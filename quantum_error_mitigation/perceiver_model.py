"""
perceiver_model.py

PyTorch implementation of the Perceiver-based QEM "prediction" model
described in Placidi et al., "Deep Learning Approaches to Quantum Error
Mitigation" (arXiv:2601.14226), Section 3.1 ("PERCEIVER", Eqs. 15-17).

Contents:
  - QEMDataset / collate_batch : wraps the arrays produced by
    `pauli_dataset.build_pauli_simulated_dataset()` and builds the model
    input X_in = [X_CB ; broadcast(Pnoisy)].
  - Perceiver                  : the model itself.
  - l1_relative_change         : the paper's performance metric (Eq. 21).
  - train_perceiver / evaluate_l1rc : training loop and evaluation helper.
  - reproduce_figure_6a        : trains a Perceiver on the Pauli Simulated
    (ibm_algiers-style) dataset and reproduces the "PERCEIVER" box of
    Figure 6a — a box plot of the L1 Relative Change on held-out test data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class QEMDataset(Dataset):
    """
    Wraps circuit (C), backend (B), noisy (Pnoisy) and ideal (Pideal) arrays.

    C:      (N, n_layers, n_qubits, 5)   -- Table 1 gate encoding
    B:      (N, 101)                     -- Section 2.2 backend calibration
    Pnoisy: (N, 32)
    Pideal: (N, 32)
    """

    def __init__(self, C, B, Pnoisy, Pideal):
        self.C = torch.as_tensor(np.asarray(C), dtype=torch.float32)
        self.B = torch.as_tensor(np.asarray(B), dtype=torch.float32)
        self.Pnoisy = torch.as_tensor(np.asarray(Pnoisy), dtype=torch.float32)
        self.Pideal = torch.as_tensor(np.asarray(Pideal), dtype=torch.float32)
        self.n_layers = self.C.shape[1]
        self.n_qubits = self.C.shape[2]

    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, idx):
        c = self.C[idx].reshape(self.n_layers, -1)  # (nl, n_qubits*5)
        b = self.B[idx]
        pnoisy = self.Pnoisy[idx]
        pideal = self.Pideal[idx]
        return c, b, pnoisy, pideal


def collate_batch(batch, feature_dim: int = 132):
    """
    Builds, for each layer, X_CB = [flattened gate encoding ; broadcast backend
    vector], zero-padded/truncated to `feature_dim` (paper: 126 real features
    padded to 132), then concatenates the broadcast noisy distribution to
    form X_in in R^{n_layers x (feature_dim + n_outcomes)} -- Section 3.1,
    "Perceiver" paragraph: "the noisy distribution is broadcast along the
    sequence length and concatenated".
    """
    cs, bs, pnoisy, pideal = zip(*batch)
    cs = torch.stack(cs)          # (B, nl, n_qubits*5)
    bs = torch.stack(bs)          # (B, 101)
    pnoisy = torch.stack(pnoisy)  # (B, n_outcomes)
    pideal = torch.stack(pideal)  # (B, n_outcomes)

    B_, nl, _ = cs.shape
    b_broadcast = bs.unsqueeze(1).expand(B_, nl, bs.shape[-1])
    xcb = torch.cat([cs, b_broadcast], dim=-1)

    if xcb.shape[-1] < feature_dim:
        pad = torch.zeros(B_, nl, feature_dim - xcb.shape[-1])
        xcb = torch.cat([xcb, pad], dim=-1)
    else:
        xcb = xcb[..., :feature_dim]

    pnoisy_broadcast = pnoisy.unsqueeze(1).expand(B_, nl, pnoisy.shape[-1])
    x_in = torch.cat([xcb, pnoisy_broadcast], dim=-1)  # (B, nl, feature_dim+n_outcomes)
    return x_in, pnoisy, pideal


# ---------------------------------------------------------------------------
# Perceiver building blocks
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Z0 = CrossAttn(queries=Z_init, keys/values=X_in)   (Eq. 15)."""

    def __init__(self, latent_dim: int, input_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(input_dim)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.kv_proj = nn.Linear(input_dim, 2 * latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, latents, x_in):
        q = self.q_proj(self.norm_q(latents))
        k, v = self.kv_proj(self.norm_kv(x_in)).chunk(2, dim=-1)
        attn_out, _ = self.attn(q, k, v)
        latents = latents + attn_out
        latents = latents + self.ff(latents)
        return latents


class LatentSelfAttentionBlock(nn.Module):
    """Z_{k+1} = LatentSelfAttnBlock(Z_k)   (Eq. 16)."""

    def __init__(self, latent_dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

    def forward(self, z):
        h = self.norm1(z)
        attn_out, _ = self.attn(h, h, h)
        z = z + attn_out
        z = z + self.ff(self.norm2(z))
        return z


class Perceiver(nn.Module):
    """
    Perceiver-based prediction model (Section 3.1, Eqs. 15-17):

        Xin        = [X_CB ; broadcast(Pnoisy)]
        Z0         = CrossAttn(queries=Z_init, keys/values=Xin)
        Z_{k+1}    = LatentSelfAttnBlock(Z_k),  k = 0 .. K-1
        Pmit       = softmax(MLP(meanpool(Z_K)))

    Default hyperparameters follow the best Pauli/Real configuration from
    Table 13 of the paper (hidden_size=1024, latent_size=256, 5 blocks,
    heads=(4,8), dropout=0.155). For the Random dataset the paper instead
    uses hidden_size=768.
    """

    def __init__(
        self,
        input_dim: int,
        n_outcomes: int = 32,
        latent_size: int = 256,
        latent_dim: int = 1024,
        n_self_attn_blocks: int = 5,
        cross_attn_heads: int = 4,
        self_attn_heads: int = 8,
        dropout: float = 0.155,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(latent_size, latent_dim) * 0.02)

        self.cross_attn = CrossAttentionBlock(latent_dim, latent_dim, cross_attn_heads, dropout)
        self.self_attn_blocks = nn.ModuleList([
            LatentSelfAttentionBlock(latent_dim, self_attn_heads, dropout)
            for _ in range(n_self_attn_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, n_outcomes),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: (B, n_layers, input_dim) -> returns log-probabilities (B, n_outcomes)."""
        batch = x_in.shape[0]
        x_proj = self.input_proj(x_in)
        z = self.latents.unsqueeze(0).expand(batch, -1, -1)
        z = self.cross_attn(z, x_proj)
        for block in self.self_attn_blocks:
            z = block(z)
        pooled = z.mean(dim=1)          # meanpool(Z_K)
        logits = self.head(pooled)
        return F.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Metric: L1 Relative Change (Eq. 21)
# ---------------------------------------------------------------------------

def l1_relative_change(p_mit: np.ndarray, p_ideal: np.ndarray, p_noisy: np.ndarray) -> np.ndarray:
    """
    R(Pmit, Pideal, Pnoisy) = (||Pideal-Pmit||_1 - ||Pideal-Pnoisy||_1) / ||Pideal-Pnoisy||_1

    Negative values indicate successful mitigation.
    """
    num = np.abs(p_ideal - p_mit).sum(axis=-1) - np.abs(p_ideal - p_noisy).sum(axis=-1)
    denom = np.abs(p_ideal - p_noisy).sum(axis=-1)
    denom = np.where(denom == 0, 1e-12, denom)
    return num / denom


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_perceiver(
    model: Perceiver,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1.25e-5,
    weight_decay: float = 9.13e-3,
    device: str = "cpu",
    patience: int = 10,
):
    """
    Trains with the KL-divergence loss (Eq. 20 / Section 3.2), AdamW,
    ReduceLROnPlateau, and early stopping -- matching the paper's
    training methodology.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for x_in, _pnoisy, pideal in train_loader:
            x_in, pideal = x_in.to(device), pideal.to(device)
            optimizer.zero_grad()
            log_pmit = model(x_in)
            loss = kldiv(log_pmit, pideal.clamp_min(1e-12))
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_in, _pnoisy, pideal in val_loader:
                x_in, pideal = x_in.to(device), pideal.to(device)
                log_pmit = model(x_in)
                val_losses.append(kldiv(log_pmit, pideal.clamp_min(1e-12)).item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_l1rc(model: Perceiver, loader: DataLoader, device: str = "cpu") -> np.ndarray:
    """Runs the model on `loader` and returns the L1 Relative Change per circuit."""
    model.eval()
    all_rc = []
    with torch.no_grad():
        for x_in, pnoisy, pideal in loader:
            x_in = x_in.to(device)
            pmit = model(x_in).exp().cpu().numpy()
            rc = l1_relative_change(pmit, pideal.numpy(), pnoisy.numpy())
            all_rc.append(rc)
    return np.concatenate(all_rc)


# ---------------------------------------------------------------------------
# Figure 6a reproduction
# ---------------------------------------------------------------------------

def reproduce_figure_6a(
    npz_path: str = "pauli_simulated_dataset.npz",
    feature_dim: int = 132,
    epochs: int = 50,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 0,
    save_path: str = "figure_6a_perceiver.png",
):
    """
    Reproduces the PERCEIVER result of Figure 6a: "Trained on Algiers Pauli
    Simulated - L1 Relative Change" (box plot; paper reports PERCEIVER
    median approx -0.57 with ~91% of circuits improved, Table 14).

    Steps (matching Section 3.2 methodology):
      1. Load the Pauli Simulated (ibm_algiers-style) dataset produced by
         pauli_dataset.build_pauli_simulated_dataset().
      2. Split into 50% train / 12.5% val / 37.5% test.
      3. Train a Perceiver ("prediction" model) with KL-divergence loss.
      4. Evaluate the L1 Relative Change (Eq. 21) on the held-out test set
         and render it as a box plot (whiskers at 1st-99th percentile),
         matching the styling of Figure 6a.
    """
    torch.manual_seed(seed)
    data = np.load(npz_path)
    C, B, Pnoisy, Pideal = data["C"], data["B"], data["Pnoisy"], data["Pideal"]

    full_ds = QEMDataset(C, B, Pnoisy, Pideal)
    n = len(full_ds)
    n_train = int(0.50 * n)
    n_val = int(0.125 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, feature_dim=feature_dim),
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    n_outcomes = Pnoisy.shape[-1]
    input_dim = feature_dim + n_outcomes
    model = Perceiver(input_dim=input_dim, n_outcomes=n_outcomes)

    model = train_perceiver(model, train_loader, val_loader, epochs=epochs, device=device)
    rc = evaluate_l1rc(model, test_loader, device=device)

    median = float(np.median(rc))
    pct_improved = 100.0 * float(np.mean(rc < 0))
    print(
        f"PERCEIVER, trained on Algiers Pauli Simulated data: "
        f"L1RC median={median:.4f}, L1RC %% improved={pct_improved:.1f}%%"
    )

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(rc, whis=(1, 99), showfliers=False, tick_labels=["PERCEIVER"])
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_ylabel("L1 Relative Change")
    ax.set_title("Trained on Algiers Pauli Simulated\n(PERCEIVER, cf. Figure 6a)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

    return rc, model


if __name__ == "__main__":
    reproduce_figure_6a()
