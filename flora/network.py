import torch
import torch.nn as nn
import torch.nn.functional as F

from flora.extract import extract_formula
from flora.ops import soft_and, soft_finally, soft_globally, soft_implies, soft_max, soft_min, soft_or


class PredicateLayer(nn.Module):
    """Learnable thresholds and scales for raw predicate signals.

    Transforms raw signals into [-1, 1] predicate values:
    P(t) = tanh(scale * (signal(t) - threshold))

    When scale is large, this becomes a sharp step function at the threshold.
    When scale is moderate, it's a smooth transition.
    """

    def __init__(self, n_predicates: int):
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(n_predicates))  # learnable threshold
        self.scale = nn.Parameter(torch.ones(n_predicates))       # learnable scale

    def forward(self, x):  # (B, N, T) -> (B, N, T)
        # threshold: (N,) -> (1, N, 1) for broadcasting
        # scale: (N,) -> (1, N, 1) for broadcasting
        th = self.threshold.unsqueeze(0).unsqueeze(-1)
        sc = self.scale.unsqueeze(0).unsqueeze(-1)
        return torch.tanh(sc * (x - th))


class TemporalLayer(nn.Module):
    """Soft-selects between {identity, G, F} per predicate, with optional negation."""

    def __init__(self, n_predicates: int, hardness: float = 100.0):
        super().__init__()
        self.n_predicates = n_predicates
        self.hardness = hardness
        self.w_sel = nn.Parameter(torch.zeros(n_predicates, 3))
        self.w_neg = nn.Parameter(torch.ones(n_predicates))

    def forward(self, x):  # (B, N, T) -> (B, N, T)
        x_id = x
        x_G = soft_globally(x, self.hardness)
        x_F = soft_finally(x, self.hardness)

        alpha = F.softmax(self.w_sel, dim=-1)  # (N, 3)
        x_out = (
            alpha[:, 0].unsqueeze(0).unsqueeze(-1) * x_id
            + alpha[:, 1].unsqueeze(0).unsqueeze(-1) * x_G
            + alpha[:, 2].unsqueeze(0).unsqueeze(-1) * x_F
        )

        neg = torch.tanh(self.w_neg)  # (N,)
        x_out = neg.unsqueeze(0).unsqueeze(-1) * x_out
        return x_out


class PropositionalLayer(nn.Module):
    """Pairs consecutive predicates with a learned connective from {and, or, implies}."""

    def __init__(self, n_pairs: int, hardness: float = 100.0):
        super().__init__()
        self.n_pairs = n_pairs
        self.hardness = hardness
        self.w_op = nn.Parameter(torch.zeros(n_pairs, 3))

    def forward(self, x):  # (B, N, T) -> (B, N//2, T)
        left = x[:, 0::2, :]
        right = x[:, 1::2, :]

        v_and = soft_and(left, right, self.hardness)
        v_or = soft_or(left, right, self.hardness)
        v_imp = soft_implies(left, right, self.hardness)

        w = F.softmax(self.w_op, dim=-1)  # (N//2, 3)
        out = (
            w[:, 0].unsqueeze(0).unsqueeze(-1) * v_and
            + w[:, 1].unsqueeze(0).unsqueeze(-1) * v_or
            + w[:, 2].unsqueeze(0).unsqueeze(-1) * v_imp
        )
        return out


class AggregationLayer(nn.Module):
    """Combines propositional outputs with a learned blend of and/or."""

    def __init__(self, hardness: float = 100.0):
        super().__init__()
        self.hardness = hardness
        self.w_agg = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # (B, M, T) -> (B, T)
        alpha = torch.sigmoid(self.w_agg)
        v_and = soft_min(x, dim=1, hardness=self.hardness)
        v_or = soft_max(x, dim=1, hardness=self.hardness)
        return alpha * v_and + (1 - alpha) * v_or


class LogicNetwork(nn.Module):
    """Full DAG: Input -> TemporalLayers -> PropositionalLayer -> AggregationLayer -> score."""

    def __init__(self, n_predicates: int, n_temporal_layers: int = 2, hardness: float = 100.0,
                 learnable_predicates: bool = False):
        super().__init__()
        self.n_predicates = n_predicates + (n_predicates % 2)  # ensure even
        self._original_n = n_predicates
        self.hardness = hardness

        self.predicate_layer = PredicateLayer(self.n_predicates) if learnable_predicates else None
        self.temporal_layers = nn.ModuleList(
            [TemporalLayer(self.n_predicates, hardness) for _ in range(n_temporal_layers)]
        )
        self.propositional = PropositionalLayer(self.n_predicates // 2, hardness)
        self.aggregation = AggregationLayer(hardness)

    def set_hardness(self, hardness: float):
        """Update hardness for all layers (used for annealing during training)."""
        self.hardness = hardness
        for layer in self.temporal_layers:
            layer.hardness = hardness
        self.propositional.hardness = hardness
        self.aggregation.hardness = hardness

    def forward(self, x):  # (B, N, T) -> (B,)
        if x.shape[1] < self.n_predicates:
            pad = torch.zeros(
                x.shape[0],
                self.n_predicates - x.shape[1],
                x.shape[2],
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)

        if self.predicate_layer is not None:
            x = self.predicate_layer(x)

        for temp_layer in self.temporal_layers:
            x = temp_layer(x)

        x = self.propositional(x)  # (B, N//2, T)
        x = self.aggregation(x)  # (B, T)
        return x[:, 0]  # score at t=0 -> (B,)

    def get_gate_values(self):
        """Extract raw gate weight values for use with extract.py."""
        sel_weights = [layer.w_sel.detach().tolist() for layer in self.temporal_layers]
        neg_weights = [layer.w_neg.detach().tolist() for layer in self.temporal_layers]
        prop_weights = self.propositional.w_op.detach().tolist()
        agg_weight = self.aggregation.w_agg.detach().item()
        pred_params = None
        if self.predicate_layer is not None:
            pred_params = {
                'threshold': self.predicate_layer.threshold.detach().tolist(),
                'scale': self.predicate_layer.scale.detach().tolist(),
            }
        return sel_weights, neg_weights, prop_weights, agg_weight, pred_params

    def extract(self, predicate_names=None):
        """Extract an interpretable sympy formula from the current weights.

        Args:
            predicate_names: Optional list of predicate name strings.
                If None, auto-generates ["P0", "P1", ..., "P{N-1}"].

        Returns:
            Simplified sympy boolean formula.
        """
        sel, neg, prop, agg, _pred = self.get_gate_values()
        if predicate_names is None:
            predicate_names = [f"P{i}" for i in range(self.n_predicates)]
        return extract_formula(sel, neg, prop, agg, predicate_names)
