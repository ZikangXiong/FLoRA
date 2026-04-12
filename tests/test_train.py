import torch

from flora import DEVICE
from flora.network import LogicNetwork
from flora.train import (
    generate_data_G_and_F,
    generate_data_G_neg_and_F,
    generate_data_GF,
    train,
)


def _classification_accuracy(net, pos_data, neg_data):
    """Compute classification accuracy: pos→positive score, neg→negative score."""
    with torch.no_grad():
        pos_scores = net(pos_data)
        neg_scores = net(neg_data)
    pos_acc = (pos_scores > 0).float().mean().item()
    neg_acc = (neg_scores < 0).float().mean().item()
    return pos_acc, neg_acc


def test_training_reduces_loss():
    """Basic sanity: loss should decrease during training."""
    torch.manual_seed(0)
    net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=10.0).to(DEVICE)
    pos, neg = generate_data_G_and_F(200, 2, 10, g_idx=0, f_idx=1)
    losses = train(net, pos, neg, lr=0.05, epochs=100, batch_size=32)
    assert losses[-1] < losses[0], "Loss should decrease"


def test_learn_G_and_F():
    """Network should learn a formula that behaves like G(P0) ∧ F(P1)."""
    torch.manual_seed(42)
    net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=10.0).to(DEVICE)
    pos, neg = generate_data_G_and_F(1000, 2, 10, g_idx=0, f_idx=1)
    train(net, pos, neg, lr=0.05, epochs=500, batch_size=64)

    # Behavioral test on held-out data
    pos_test, neg_test = generate_data_G_and_F(200, 2, 10, g_idx=0, f_idx=1)
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.8, f"Positive accuracy too low: {pos_acc:.2f}"
    assert neg_acc > 0.8, f"Negative accuracy too low: {neg_acc:.2f}"

    # Structural check: formula should use G on P0
    formula = net.extract(["P0", "P1"])
    print(f"Learned formula: {formula}")


def test_learn_G_neg_and_F():
    """Network should learn a formula that behaves like G(~P0) ∧ F(P1)."""
    torch.manual_seed(42)
    net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=10.0).to(DEVICE)
    pos, neg = generate_data_G_neg_and_F(1000, 2, 10, g_neg_idx=0, f_idx=1)
    train(net, pos, neg, lr=0.05, epochs=500, batch_size=64)

    pos_test, neg_test = generate_data_G_neg_and_F(200, 2, 10, g_neg_idx=0, f_idx=1)
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.8, f"Positive accuracy too low: {pos_acc:.2f}"
    assert neg_acc > 0.8, f"Negative accuracy too low: {neg_acc:.2f}"

    formula = net.extract(["P0", "P1"])
    print(f"Learned formula: {formula}")


def test_learn_GF_nested():
    """Network should learn a formula that behaves like G(F(P0)) ∧ F(P1)."""
    torch.manual_seed(42)
    T = 15
    net = LogicNetwork(n_predicates=2, n_temporal_layers=2, hardness=10.0).to(DEVICE)
    pos, neg = generate_data_GF(1000, 2, T, gf_idx=0, f_idx=1)
    train(net, pos, neg, lr=0.05, epochs=500, batch_size=64)

    pos_test, neg_test = generate_data_GF(200, 2, T, gf_idx=0, f_idx=1)
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.8, f"Positive accuracy too low: {pos_acc:.2f}"
    assert neg_acc > 0.8, f"Negative accuracy too low: {neg_acc:.2f}"

    formula = net.extract(["P0", "P1"])
    print(f"Learned formula: {formula}")


def test_hardness_annealing():
    """Training with hardness annealing: start soft, end crisp."""
    torch.manual_seed(42)
    net = LogicNetwork(n_predicates=2, n_temporal_layers=1, hardness=5.0).to(DEVICE)
    pos, neg = generate_data_G_and_F(500, 2, 10, g_idx=0, f_idx=1)
    losses = train(net, pos, neg, lr=0.05, epochs=200, batch_size=32,
                   anneal_hardness=(5.0, 50.0))
    assert losses[-1] < losses[0]
    assert abs(net.hardness - 50.0) < 0.1

    pos_test, neg_test = generate_data_G_and_F(200, 2, 10, g_idx=0, f_idx=1)
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.7, f"Positive accuracy: {pos_acc:.2f}"
    assert neg_acc > 0.7, f"Negative accuracy: {neg_acc:.2f}"


def test_learn_implication():
    """Learn G(P0) -> F(P1): if P0 always holds, then P1 must eventually hold.

    Positive data has two types (both satisfy the implication):
      (a) P0 always positive AND P1 eventually positive
      (b) P0 has negative dip (antecedent false -> vacuously true)
    Negative: P0 always positive BUT P1 always negative.
    """
    torch.manual_seed(42)
    T, N, n = 10, 2, 500

    # Positive type (a): G(P0) true, F(P1) true
    pos_a = torch.rand(n, N, T, device=DEVICE) * 2 - 1
    pos_a[:, 0, :] = torch.abs(torch.randn(n, T, device=DEVICE)) * 0.3 + 0.2
    pos_a[:, 1, :] = -torch.abs(torch.randn(n, T, device=DEVICE)) * 0.3 - 0.1
    for i in range(n):
        pos_a[i, 1, torch.randint(0, T, (1,))] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    # Positive type (b): G(P0) false -> vacuously true
    pos_b = torch.rand(n, N, T, device=DEVICE) * 2 - 1
    pos_b[:, 0, :] = torch.abs(torch.randn(n, T, device=DEVICE)) * 0.3 + 0.2
    for i in range(n):
        pos_b[i, 0, torch.randint(0, T, (1,))] = -torch.abs(torch.randn(1)).item() * 0.3 - 0.3
    pos_b[:, 1, :] = torch.randn(n, T, device=DEVICE) * 0.5

    pos = torch.cat([pos_a, pos_b], dim=0)

    # Negative: G(P0) true BUT F(P1) false
    neg = torch.rand(n, N, T, device=DEVICE) * 2 - 1
    neg[:, 0, :] = torch.abs(torch.randn(n, T, device=DEVICE)) * 0.3 + 0.2
    neg[:, 1, :] = -torch.abs(torch.randn(n, T, device=DEVICE)) * 0.3 - 0.2

    net = LogicNetwork(n_predicates=N, n_temporal_layers=1, hardness=10.0).to(DEVICE)
    train(net, pos, neg, lr=0.05, epochs=500, batch_size=64)

    # Test on type (a) positive and negative
    pos_test = pos_a[:100]
    neg_test = neg[:100]
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.7, f"Positive accuracy: {pos_acc:.2f}"
    assert neg_acc > 0.7, f"Negative accuracy: {neg_acc:.2f}"

    formula = net.extract(["P0", "P1"])
    print(f"Implication formula: {formula}")
