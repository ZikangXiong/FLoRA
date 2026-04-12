import torch
import torch.nn.functional as F
from torch import Tensor


def soft_min(x: Tensor, dim: int, hardness: float = 100.0) -> Tensor:
    """Differentiable approximation of min.

    softmin(x) = sum(x * softmax(-x * hardness, dim))
    As hardness -> inf, approaches true min.
    """
    weights = F.softmax(-x * hardness, dim=dim)
    return (x * weights).sum(dim=dim)


def soft_max(x: Tensor, dim: int, hardness: float = 100.0) -> Tensor:
    """Differentiable approximation of max.

    softmax(x) = sum(x * softmax(x * hardness, dim))
    As hardness -> inf, approaches true max.
    """
    weights = F.softmax(x * hardness, dim=dim)
    return (x * weights).sum(dim=dim)


def soft_globally(x: Tensor, hardness: float = 100.0) -> Tensor:
    """G(phi)(t) ~ min_{t'>=t} phi(t') for all t simultaneously.

    Input: (B, N, T) or (..., T)
    Output: same shape, suffix-min at each position.

    Uses an upper-triangular (causal-future) mask for vectorized computation.
    """
    T = x.shape[-1]
    # Future mask: mask[t, t'] = True if t' >= t (upper triangular)
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
    # Large negative for masked positions
    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)

    # x_expanded: (..., 1, T), scores over the last T dim for each query t
    x_expanded = x.unsqueeze(-2).expand(*x.shape[:-1], T, T)
    # Apply mask: for query position t, only positions t' >= t are valid
    logits = torch.where(mask, -x_expanded * hardness, neg_inf)
    weights = F.softmax(logits, dim=-1)
    # Weighted sum: out[..., t] = sum_{t'>=t} w[t,t'] * x[t']
    out = (weights * x_expanded).sum(dim=-1)
    return out


def soft_finally(x: Tensor, hardness: float = 100.0) -> Tensor:
    """F(phi)(t) ~ max_{t'>=t} phi(t') for all t simultaneously.

    Input: (B, N, T) or (..., T)
    Output: same shape, suffix-max at each position.

    Uses an upper-triangular (causal-future) mask for vectorized computation.
    """
    T = x.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
    neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)

    x_expanded = x.unsqueeze(-2).expand(*x.shape[:-1], T, T)
    logits = torch.where(mask, x_expanded * hardness, neg_inf)
    weights = F.softmax(logits, dim=-1)
    out = (weights * x_expanded).sum(dim=-1)
    return out


def hard_globally(x: Tensor) -> Tensor:
    """True suffix min: G(phi)(t) = min_{t'>=t} phi(t')."""
    return torch.flip(torch.cummin(torch.flip(x, [-1]), dim=-1).values, [-1])


def hard_finally(x: Tensor) -> Tensor:
    """True suffix max: F(phi)(t) = max_{t'>=t} phi(t')."""
    return torch.flip(torch.cummax(torch.flip(x, [-1]), dim=-1).values, [-1])


def soft_and(a: Tensor, b: Tensor, hardness: float = 100.0) -> Tensor:
    """Soft min(a, b)."""
    return soft_min(torch.stack([a, b], dim=-1), dim=-1, hardness=hardness)


def soft_or(a: Tensor, b: Tensor, hardness: float = 100.0) -> Tensor:
    """Soft max(a, b)."""
    return soft_max(torch.stack([a, b], dim=-1), dim=-1, hardness=hardness)


def soft_not(a: Tensor) -> Tensor:
    """Negation: -a."""
    return -a


def soft_implies(a: Tensor, b: Tensor, hardness: float = 100.0) -> Tensor:
    """a -> b = max(-a, b) — soft version."""
    return soft_or(soft_not(a), b, hardness=hardness)
