import torch
from torch import Tensor

from flora import DEVICE
from flora.network import LogicNetwork


def generate_data_G_and_F(
    n_samples: int, n_predicates: int, T: int, g_idx: int, f_idx: int
) -> tuple[Tensor, Tensor]:
    """Generate positive and negative data for G(P_g) ∧ F(P_f).

    Positive: P_g always positive, P_f eventually positive.
    Negative: P_g has a negative dip OR P_f always negative.
    """
    # --- Positive ---
    pos = torch.rand(n_samples, n_predicates, T, device=DEVICE) * 2 - 1
    pos[:, g_idx, :] = torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 + 0.2
    pos[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        pos[i, f_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    # --- Negative (violates G(P_g)): P_g has a negative dip ---
    neg = torch.rand(n_samples, n_predicates, T, device=DEVICE) * 2 - 1
    neg[:, g_idx, :] = torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 + 0.2
    dip_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        neg[i, g_idx, dip_t[i]] = -torch.abs(torch.randn(1)).item() * 0.3 - 0.2
    # P_f still eventually positive (so only G violation matters)
    neg[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        neg[i, f_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    return pos, neg


def generate_data_G_neg_and_F(
    n_samples: int, n_predicates: int, T: int, g_neg_idx: int, f_idx: int
) -> tuple[Tensor, Tensor]:
    """Generate data for G(~P_i) ∧ F(P_j).

    Positive: P_i always negative, P_j eventually positive.
    Negative: P_i has a positive spike (violates G(~P_i)).
    """
    # --- Positive ---
    pos = torch.rand(n_samples, n_predicates, T, device=DEVICE) * 2 - 1
    pos[:, g_neg_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.2
    pos[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        pos[i, f_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    # --- Negative (violates G(~P_i)): P_i has a positive spike ---
    neg = torch.rand(n_samples, n_predicates, T, device=DEVICE) * 2 - 1
    neg[:, g_neg_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.2
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        neg[i, g_neg_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.2
    neg[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t2 = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        neg[i, f_idx, spike_t2[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    return pos, neg


def generate_data_GF(
    n_samples: int, n_predicates: int, T: int, gf_idx: int, f_idx: int
) -> tuple[Tensor, Tensor]:
    """Generate data for G(F(P_i)) ∧ F(P_j).

    Positive: P_i has recurring positive spikes (every suffix has a positive),
              P_j eventually positive.
    Negative: P_i has no positive values in some suffix (violates G(F(P_i))).
    """
    # --- Positive: recurring spikes ---
    pos = torch.randn(n_samples, n_predicates, T, device=DEVICE) * 0.3 - 0.2
    window = max(T // 4, 2)
    for start in range(0, T, window):
        end = min(start + window, T)
        spike_t = torch.randint(start, end, (n_samples,))
        for i in range(n_samples):
            pos[i, gf_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.4
    # P_j eventually positive
    pos[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        pos[i, f_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    # --- Negative: P_i has a dead zone (no positive in last portion) ---
    neg = torch.randn(n_samples, n_predicates, T, device=DEVICE) * 0.3 - 0.2
    # Put spikes only in first half, leave second half negative
    for start in range(0, T // 2, window):
        end = min(start + window, T // 2)
        spike_t = torch.randint(start, end, (n_samples,))
        for i in range(n_samples):
            neg[i, gf_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.4
    neg[:, gf_idx, T // 2 :] = -torch.abs(
        torch.randn(n_samples, T - T // 2, device=DEVICE)
    ) * 0.3 - 0.3
    # P_j still eventually positive
    neg[:, f_idx, :] = -torch.abs(torch.randn(n_samples, T, device=DEVICE)) * 0.3 - 0.1
    spike_t = torch.randint(0, T, (n_samples,))
    for i in range(n_samples):
        neg[i, f_idx, spike_t[i]] = torch.abs(torch.randn(1)).item() * 0.3 + 0.3

    return pos, neg


def train(
    net: LogicNetwork,
    pos_data: Tensor,
    neg_data: Tensor | None = None,
    lr: float = 0.05,
    epochs: int = 200,
    batch_size: int = 32,
    margin: float = 0.1,
    rho: float = 0.1,
    anneal_hardness: tuple[float, float] | None = None,
    verbose: bool = False,
) -> list[float]:
    """Train with contrastive loss: push positive scores up, negative scores down.

    Loss = relu(margin - pos_score) + relu(neg_score + margin) + rho * agg_reg

    Args:
        anneal_hardness: If provided, (start, end) hardness values. Linearly
            anneals from start to end over training. Low start = smooth gradients,
            high end = crisp operators for extraction.
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []

    n_pos = pos_data.shape[0]
    n_neg = neg_data.shape[0] if neg_data is not None else 0

    for epoch in range(epochs):
        if anneal_hardness is not None:
            h_start, h_end = anneal_hardness
            t = epoch / max(epochs - 1, 1)
            net.set_hardness(h_start + t * (h_end - h_start))

        perm_pos = torch.randperm(n_pos, device=pos_data.device)
        perm_neg = torch.randperm(n_neg, device=pos_data.device) if n_neg > 0 else None
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_pos, batch_size):
            pos_batch = pos_data[perm_pos[start : start + batch_size]]
            pos_scores = net(pos_batch)
            loss = torch.relu(margin - pos_scores).mean()

            if neg_data is not None and perm_neg is not None:
                neg_start = start % n_neg
                neg_batch = neg_data[
                    perm_neg[neg_start : neg_start + batch_size]
                ]
                neg_scores = net(neg_batch)
                loss = loss + torch.relu(neg_scores + margin).mean()

            loss = loss + rho * torch.relu(-net.aggregation.w_agg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 50 == 0:
            h_str = f", hardness={net.hardness:.1f}" if anneal_hardness else ""
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}{h_str}")

    return losses
