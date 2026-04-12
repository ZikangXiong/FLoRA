import torch
import pytest

from flora.ops import (
    hard_finally,
    hard_globally,
    soft_and,
    soft_finally,
    soft_globally,
    soft_implies,
    soft_min,
    soft_max,
    soft_not,
    soft_or,
)


@pytest.fixture
def sample_input():
    """(B=2, N=3, T=5) random tensor."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 5)


# --- Shape preservation ---


def test_soft_globally_shape(sample_input):
    out = soft_globally(sample_input)
    assert out.shape == sample_input.shape


def test_soft_finally_shape(sample_input):
    out = soft_finally(sample_input)
    assert out.shape == sample_input.shape


def test_hard_globally_shape(sample_input):
    out = hard_globally(sample_input)
    assert out.shape == sample_input.shape


def test_hard_finally_shape(sample_input):
    out = hard_finally(sample_input)
    assert out.shape == sample_input.shape


# --- Convergence to hard operators ---


def test_soft_globally_converges_to_hard(sample_input):
    hard = hard_globally(sample_input)
    soft = soft_globally(sample_input, hardness=1000.0)
    torch.testing.assert_close(soft, hard, atol=1e-3, rtol=1e-3)


def test_soft_finally_converges_to_hard(sample_input):
    hard = hard_finally(sample_input)
    soft = soft_finally(sample_input, hardness=1000.0)
    torch.testing.assert_close(soft, hard, atol=1e-3, rtol=1e-3)


# --- Monotonicity ---


def test_soft_globally_monotonicity(sample_input):
    """soft_globally(x)[t] <= x[t] for all t (min over suffix including x[t])."""
    out = soft_globally(sample_input)
    assert (out <= sample_input + 1e-5).all()


def test_soft_finally_monotonicity(sample_input):
    """soft_finally(x)[t] >= x[t] for all t (max over suffix including x[t])."""
    out = soft_finally(sample_input)
    assert (out >= sample_input - 1e-5).all()


# --- Boundary (last timestep) ---


def test_soft_globally_boundary(sample_input):
    """At last timestep, suffix has length 1, so output ~ input."""
    out = soft_globally(sample_input)
    torch.testing.assert_close(out[..., -1], sample_input[..., -1], atol=1e-5, rtol=1e-5)


def test_soft_finally_boundary(sample_input):
    out = soft_finally(sample_input)
    torch.testing.assert_close(out[..., -1], sample_input[..., -1], atol=1e-5, rtol=1e-5)


# --- Known values ---


def test_hard_globally_known():
    x = torch.tensor([[[0.5, -0.3, 0.8]]])  # (1, 1, 3)
    expected = torch.tensor([[[-0.3, -0.3, 0.8]]])
    torch.testing.assert_close(hard_globally(x), expected)


def test_hard_finally_known():
    x = torch.tensor([[[0.5, -0.3, 0.8]]])
    expected = torch.tensor([[[0.8, 0.8, 0.8]]])
    torch.testing.assert_close(hard_finally(x), expected)


def test_soft_globally_known():
    x = torch.tensor([[[0.5, -0.3, 0.8]]])
    out = soft_globally(x, hardness=1000.0)
    expected = torch.tensor([[[-0.3, -0.3, 0.8]]])
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


# --- Gradient flow ---


def test_soft_globally_gradient_flow():
    """All input positions should receive non-zero gradient."""
    x = torch.tensor([[[0.5, -0.3, 0.8]]], requires_grad=True)
    # Use lower hardness so gradients don't vanish for non-min elements.
    out = soft_globally(x, hardness=1.0)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad != 0).all(), f"Some gradients are zero: {x.grad}"


def test_soft_finally_gradient_flow():
    x = torch.tensor([[[0.5, -0.3, 0.8]]], requires_grad=True)
    # Use lower hardness so gradients don't vanish for non-max elements.
    out = soft_finally(x, hardness=1.0)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad != 0).all(), f"Some gradients are zero: {x.grad}"


# --- Logical connectives ---


def test_soft_and():
    a = torch.tensor(0.5)
    b = torch.tensor(-0.3)
    out = soft_and(a, b, hardness=1000.0)
    torch.testing.assert_close(out, b, atol=1e-3, rtol=1e-3)


def test_soft_or():
    a = torch.tensor(0.5)
    b = torch.tensor(-0.3)
    out = soft_or(a, b, hardness=1000.0)
    torch.testing.assert_close(out, a, atol=1e-3, rtol=1e-3)


def test_soft_not():
    a = torch.tensor(0.5)
    assert soft_not(a).item() == pytest.approx(-0.5)


def test_soft_implies():
    a = torch.tensor(0.5)
    b = torch.tensor(-0.3)
    # a -> b = max(-a, b) = max(-0.5, -0.3) = -0.3
    out = soft_implies(a, b, hardness=1000.0)
    torch.testing.assert_close(out, torch.tensor(-0.3), atol=1e-3, rtol=1e-3)


def test_soft_implies_true_antecedent():
    """When a is very negative (false), implies should be ~true (high value)."""
    a = torch.tensor(-2.0)
    b = torch.tensor(-1.0)
    # a -> b = max(-a, b) = max(2.0, -1.0) = 2.0
    out = soft_implies(a, b, hardness=1000.0)
    torch.testing.assert_close(out, torch.tensor(2.0), atol=1e-3, rtol=1e-3)


# --- soft_min / soft_max basic ---


def test_soft_min_basic():
    x = torch.tensor([3.0, 1.0, 2.0])
    out = soft_min(x, dim=0, hardness=1000.0)
    assert out.item() == pytest.approx(1.0, abs=1e-3)


def test_soft_max_basic():
    x = torch.tensor([3.0, 1.0, 2.0])
    out = soft_max(x, dim=0, hardness=1000.0)
    assert out.item() == pytest.approx(3.0, abs=1e-3)
