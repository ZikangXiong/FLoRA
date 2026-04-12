import torch
import pytest

from flora.network import (
    AggregationLayer,
    LogicNetwork,
    PredicateLayer,
    PropositionalLayer,
    TemporalLayer,
)
from flora.ops import soft_and, soft_finally


# --- Shape tests ---


def test_logic_network_output_shape():
    net = LogicNetwork(4, 2)
    x = torch.randn(8, 4, 10)
    out = net(x)
    assert out.shape == (8,)


def test_logic_network_odd_predicates_shape():
    net = LogicNetwork(5, 2)
    x = torch.randn(8, 5, 10)
    out = net(x)
    assert out.shape == (8,)


def test_temporal_layer_shape():
    layer = TemporalLayer(4)
    x = torch.randn(8, 4, 10)
    out = layer(x)
    assert out.shape == (8, 4, 10)


def test_propositional_layer_shape():
    layer = PropositionalLayer(2)
    x = torch.randn(8, 4, 10)
    out = layer(x)
    assert out.shape == (8, 2, 10)


def test_aggregation_layer_shape():
    layer = AggregationLayer()
    x = torch.randn(8, 3, 10)
    out = layer(x)
    assert out.shape == (8, 10)


# --- Gradient flow ---


def test_gradient_flow():
    net = LogicNetwork(4)
    x = torch.randn(8, 4, 10)
    loss = -net(x).mean()
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.all(param.grad == 0), f"{name} gradient is all zeros"


# --- get_gate_values ---


def test_get_gate_values_shapes():
    n_predicates = 4
    n_temporal = 2
    net = LogicNetwork(n_predicates, n_temporal)
    sel, neg, prop, agg, pred = net.get_gate_values()

    # sel: K lists, each N x 3
    assert len(sel) == n_temporal
    for s in sel:
        assert len(s) == net.n_predicates
        for row in s:
            assert len(row) == 3

    # neg: K lists, each N
    assert len(neg) == n_temporal
    for n in neg:
        assert len(n) == net.n_predicates

    # prop: N//2 x 3
    assert len(prop) == net.n_predicates // 2
    for row in prop:
        assert len(row) == 3

    # agg: scalar
    assert isinstance(agg, float)


# --- Deterministic with known weights ---


def test_deterministic_known_weights():
    torch.manual_seed(0)
    net = LogicNetwork(2, n_temporal_layers=1, hardness=100.0)

    with torch.no_grad():
        # Predicate 0: identity, Predicate 1: F
        net.temporal_layers[0].w_sel.copy_(
            torch.tensor([[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        )
        # No negation
        net.temporal_layers[0].w_neg.copy_(torch.tensor([1.0, 1.0]))
        # Propositional: and
        net.propositional.w_op.copy_(torch.tensor([[10.0, 0.0, 0.0]]))
        # Aggregation: doesn't matter with 1 pair, but set to and
        net.aggregation.w_agg.data.fill_(10.0)

    x = torch.randn(4, 2, 6)
    out = net(x)

    # Expected: soft_and(tanh(1)*x[:,0,:], tanh(1)*soft_finally(x[:,1,:]))
    # then take t=0
    neg = torch.tanh(torch.tensor(1.0))
    left = neg * x[:, 0, :]  # (4, 6) - identity applied
    right = neg * soft_finally(x[:, 1:2, :], 100.0).squeeze(1)  # (4, 6)
    expected = soft_and(left, right, 100.0)[:, 0]  # score at t=0

    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


# --- Odd predicate padding ---


def test_odd_predicate_padding_count():
    net = LogicNetwork(3)
    assert net.n_predicates == 4


def test_odd_predicate_padding_forward():
    net = LogicNetwork(3)
    x = torch.randn(4, 3, 8)
    out = net(x)
    assert out.shape == (4,)


# --- extract method ---


def test_extract_method():
    """LogicNetwork.extract() should return a sympy formula."""
    net = LogicNetwork(n_predicates=4, n_temporal_layers=2, hardness=10.0)
    formula = net.extract()
    assert formula is not None
    formula_str = str(formula)
    # With default names, should contain P0, P1, etc.
    assert "P0" in formula_str or "P1" in formula_str


def test_extract_method_custom_names():
    net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=10.0)
    formula = net.extract(predicate_names=["safe", "goal"])
    formula_str = str(formula)
    assert "safe" in formula_str or "goal" in formula_str


# --- PredicateLayer tests ---


def test_predicate_layer_shape():
    layer = PredicateLayer(4)
    x = torch.randn(8, 4, 10)
    out = layer(x)
    assert out.shape == (8, 4, 10)
    # Output should be in [-1, 1]
    assert out.min() >= -1.0
    assert out.max() <= 1.0


def test_logic_network_with_learnable_predicates():
    net = LogicNetwork(n_predicates=4, n_temporal_layers=2, hardness=10.0, learnable_predicates=True)
    x = torch.randn(8, 4, 10)
    scores = net(x)
    assert scores.shape == (8,)


def test_get_gate_values_with_predicates():
    net = LogicNetwork(n_predicates=4, learnable_predicates=True)
    sel, neg, prop, agg, pred = net.get_gate_values()
    assert pred is not None
    assert len(pred['threshold']) == 4
    assert len(pred['scale']) == 4


def test_get_gate_values_without_predicates():
    net = LogicNetwork(n_predicates=4)
    sel, neg, prop, agg, pred = net.get_gate_values()
    assert pred is None
