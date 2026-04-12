import time

import torch

from flora import DEVICE
from flora.network import LogicNetwork
from flora.train import train


def _classification_accuracy(net, pos_data, neg_data):
    """Compute classification accuracy: pos->positive score, neg->negative score."""
    with torch.no_grad():
        pos_scores = net(pos_data)
        neg_scores = net(neg_data)
    pos_acc = (pos_scores > 0).float().mean().item()
    neg_acc = (neg_scores < 0).float().mean().item()
    return pos_acc, neg_acc


def _generate_six_pred_data(n_samples, N, T):
    """Generate data for G(P0) & F(P1) & G(P2) & F(P3) & G(P4) & F(P5).

    Positive: even indices (0,2,4) always positive, odd (1,3,5) eventually positive.
    Negative: same pattern but ONE of {0,2,4} has a negative dip per sample.
    """
    pos = torch.rand(n_samples, N, T, device=DEVICE) * 2 - 1
    neg = torch.rand(n_samples, N, T, device=DEVICE) * 2 - 1

    # Positive data
    for i in [0, 2, 4]:
        pos[:, i, :] = torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 + 0.2
    for i in [1, 3, 5]:
        pos[:, i, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
        spike_t = torch.randint(0, T, (n_samples,))
        for j in range(n_samples):
            pos[j, i, spike_t[j]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    # Negative data: same as positive but ONE of {0,2,4} has a dip
    for i in [0, 2, 4]:
        neg[:, i, :] = torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 + 0.2
    for i in [1, 3, 5]:
        neg[:, i, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
        spike_t = torch.randint(0, T, (n_samples,))
        for j in range(n_samples):
            neg[j, i, spike_t[j]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3
    # Add dip to a random G-predicate per sample
    dip_pred = torch.randint(0, 3, (n_samples,)) * 2  # randomly 0, 2, or 4
    dip_t = torch.randint(0, T, (n_samples,))
    for j in range(n_samples):
        neg[j, dip_pred[j], dip_t[j]] = -torch.abs(torch.randn(1)).item() * 0.3 - 0.2

    return pos, neg


def test_six_predicates():
    """Train on G(P0) & F(P1) & G(P2) & F(P3) & G(P4) & F(P5) with N=6."""
    torch.manual_seed(42)
    T = 10
    N = 6
    net = LogicNetwork(n_predicates=N, n_temporal_layers=1, hardness=10.0).to(DEVICE)

    pos, neg = _generate_six_pred_data(1000, N, T)
    train(net, pos, neg, lr=0.05, epochs=500, batch_size=64)

    # Behavioral test on fresh data
    pos_test, neg_test = _generate_six_pred_data(200, N, T)
    pos_acc, neg_acc = _classification_accuracy(net, pos_test, neg_test)
    assert pos_acc > 0.7, f"Positive accuracy too low: {pos_acc:.2f}"
    assert neg_acc > 0.7, f"Negative accuracy too low: {neg_acc:.2f}"

    # Print learned formula
    formula = net.extract([f"P{i}" for i in range(N)])
    print(f"N=6 formula: {formula}")


def test_ten_predicates_forward():
    """Verify LogicNetwork handles 10 predicates: forward + backward."""
    net = LogicNetwork(n_predicates=10, n_temporal_layers=2, hardness=10.0).to(DEVICE)
    x = torch.randn(16, 10, 20, device=DEVICE)
    scores = net(x)
    assert scores.shape == (16,)

    loss = -scores.mean()
    loss.backward()
    for p in net.parameters():
        assert p.grad is not None

    formula = net.extract()
    print(f"N=10 (untrained) formula: {formula}")


def test_ten_predicates_with_learnable_predicates():
    """Verify forward + backward works with PredicateLayer at N=10."""
    net = LogicNetwork(
        n_predicates=10, n_temporal_layers=2, hardness=10.0, learnable_predicates=True
    ).to(DEVICE)
    x = torch.randn(16, 10, 20, device=DEVICE)
    scores = net(x)
    assert scores.shape == (16,)
    loss = -scores.mean()
    loss.backward()
    # Check predicate layer gets gradients too
    assert net.predicate_layer.threshold.grad is not None
    assert net.predicate_layer.scale.grad is not None


def test_timing_various_sizes():
    """Print timing for different predicate counts. Informational only."""
    for N in [4, 10, 20, 40]:
        net = LogicNetwork(n_predicates=N, n_temporal_layers=2, hardness=10.0).to(DEVICE)
        x = torch.randn(32, N, 20, device=DEVICE)

        # Warmup
        _ = net(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(10):
            scores = net(x)
            loss = -scores.mean()
            loss.backward()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        print(f"N={N:3d}: {elapsed*1000:.1f} ms/iter")
