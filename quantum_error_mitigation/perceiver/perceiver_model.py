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
import torch.utils.checkpoint as torch_checkpoint
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import gc
import tempfile


# ---------------------------------------------------------------------------
# Device / CPU-thread configuration helpers
# ---------------------------------------------------------------------------

def configure_cpu_threads(n_threads: int = None) -> int:
    """
    Explicitly sets the number of CPU threads PyTorch uses for intra-op
    parallelism (matmuls, etc). Worth calling explicitly and *verifying* --
    on some machines/containers PyTorch (or an inherited OMP_NUM_THREADS=1
    environment variable) ends up using far fewer threads than physically
    available, silently leaving most of a multi-core CPU idle during
    training. Returns the thread count actually in effect afterwards.
    """
    if n_threads is None:
        n_threads = os.cpu_count() or 1
    torch.set_num_threads(n_threads)
    return torch.get_num_threads()


def pick_device(preferred: str = "auto") -> str:
    """Returns 'cuda' if available and requested/auto, else 'cpu'."""
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def recommended_training_config(device: str = "auto") -> dict:
    """
    Suggested (batch_size, gradient_accumulation_steps, use_checkpoint,
    use_amp) starting point for a given device. These are starting points,
    not hard requirements -- tune batch_size upward on a GPU with more VRAM,
    or downward if you hit an out-of-memory error.

    Rationale:
      - CPU, ample RAM (e.g. 16GB): checkpointing trades ~50-60% extra
        compute for roughly half the peak memory (see Perceiver/
        train_perceiver docstrings) -- worth it only when RAM, not time, is
        the binding constraint. With 16GB free, it usually isn't, so this
        preset turns checkpointing off and raises the batch size instead.
      - CUDA (e.g. Colab T4, 15GB VRAM): checkpointing is essentially never
        worth it (VRAM is more comfortable and every recomputation costs
        wall-clock time you're specifically trying to save); mixed
        precision (use_amp=True) roughly halves compute time again on
        tensor-core GPUs with negligible accuracy impact.
    """
    device = pick_device(device)
    if device == "cuda":
        return dict(batch_size=128, gradient_accumulation_steps=1,
                    use_checkpoint=False, use_amp=True)
    else:
        return dict(batch_size=32, gradient_accumulation_steps=4,
                    use_checkpoint=False, use_amp=False)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class QEMDataset(Dataset):
    """
    Wraps circuit (C), backend (B), noisy (Pnoisy) and ideal (Pideal) arrays.

    Parameters
    ----------
    subset_fraction : float, optional
        Fraction of the dataset to expose. Must lie in (0, 1].
        If None (default), the full dataset is used.

    seed : int
        Random seed used when sampling the subset.
    """

    def __init__(
        self,
        C,
        B,
        Pnoisy,
        Pideal,
        subset_fraction: float | None = None,
        seed: int = 0,
    ):
        self.C = torch.as_tensor(np.asarray(C), dtype=torch.float32)
        self.B = torch.as_tensor(np.asarray(B), dtype=torch.float32)
        self.Pnoisy = torch.as_tensor(np.asarray(Pnoisy), dtype=torch.float32)
        self.Pideal = torch.as_tensor(np.asarray(Pideal), dtype=torch.float32)

        self.n_layers = self.C.shape[1]
        self.n_qubits = self.C.shape[2]

        # -------------------------------------------------------
        # Optional dataset subsampling
        # -------------------------------------------------------
        if subset_fraction is None:
            self.indices = torch.arange(self.C.shape[0])

        else:
            if not (0 < subset_fraction <= 1):
                raise ValueError(
                    "subset_fraction must lie in the interval (0, 1]."
                )

            rng = np.random.default_rng(seed)

            n_total = self.C.shape[0]
            n_subset = max(1, int(round(subset_fraction * n_total)))

            self.indices = torch.as_tensor(
                rng.choice(n_total, size=n_subset, replace=False),
                dtype=torch.long,
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        c = self.C[idx].reshape(self.n_layers, -1)
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
# Memory-mapped dataset variant
#
# QEMDataset above loads the *entire* C/B/Pnoisy/Pideal arrays into RAM as
# torch tensors. For the paper's full scale (~144,000 circuit-repeat
# samples), C alone (padded circuit encodings) can run into several GB,
# which does not fit comfortably alongside the model, optimizer state, and
# OS overhead on a 16GB machine. LazyQEMDataset instead memory-maps each
# array from disk; the OS pages in only the samples a given batch actually
# touches, so resident memory stays close to
# O(batch_size x sample_size) rather than O(dataset_size x sample_size).
#
# True mmap only works on *uncompressed* arrays -- a compressed .npz must be
# fully decompressed into RAM to be read at all, so use save_dataset_npy()
# below (uncompressed, one file per array) rather than np.savez_compressed
# when you intend to train with LazyQEMDataset.
# ---------------------------------------------------------------------------

def save_dataset_npy(dataset: dict, directory: str) -> None:
    """
    Saves a dataset dict (as returned by
    pauli_dataset.build_pauli_simulated_dataset) as individual, uncompressed
    .npy files under `directory`, suitable for memory-mapped loading via
    LazyQEMDataset.
    """
    os.makedirs(directory, exist_ok=True)
    for key in ("C", "B", "Pnoisy", "Pideal"):
        np.save(
            os.path.join(directory, f"{key}.npy"),
            np.asarray(dataset[key], dtype=np.float32),
        )


class LazyQEMDataset(Dataset):
    """
    Memory-mapped counterpart to QEMDataset, for datasets too large to hold
    comfortably in RAM. Expects `directory` to contain C.npy, B.npy,
    Pnoisy.npy, Pideal.npy as saved by `save_dataset_npy`.

    Functionally identical to QEMDataset (same __getitem__ return shapes,
    so it is a drop-in replacement anywhere a QEMDataset is used) -- only
    the backing storage differs.
    """

    def __init__(self, directory: str):
        self.C = np.load(os.path.join(directory, "C.npy"), mmap_mode="r")
        self.B = np.load(os.path.join(directory, "B.npy"), mmap_mode="r")
        self.Pnoisy = np.load(os.path.join(directory, "Pnoisy.npy"), mmap_mode="r")
        self.Pideal = np.load(os.path.join(directory, "Pideal.npy"), mmap_mode="r")
        self.n_layers = self.C.shape[1]
        self.n_qubits = self.C.shape[2]

    def __len__(self):
        return self.C.shape[0]

    def __getitem__(self, idx):
        # np.array(...) materializes only this one sample (a small copy out
        # of the memory-mapped file), not the whole underlying array.
        c = torch.from_numpy(np.array(self.C[idx], dtype=np.float32)).reshape(self.n_layers, -1)
        b = torch.from_numpy(np.array(self.B[idx], dtype=np.float32))
        pnoisy = torch.from_numpy(np.array(self.Pnoisy[idx], dtype=np.float32))
        pideal = torch.from_numpy(np.array(self.Pideal[idx], dtype=np.float32))
        return c, b, pnoisy, pideal


def count_parameters(model: nn.Module) -> dict:
    """
    Returns a breakdown of parameter counts (and approximate fp32 memory in
    MB) per top-level submodule, plus the total. Useful for sizing a model
    configuration to a memory budget before launching a full training run.
    """
    breakdown = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        breakdown[key] = breakdown.get(key, 0) + p.numel()
    breakdown["TOTAL"] = sum(v for k, v in breakdown.items() if k != "TOTAL")
    return {k: {"params": v, "mb_fp32": v * 4 / 1e6} for k, v in breakdown.items()}


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
        # need_weights=False: we never use the returned attention weights,
        # and requesting them forces PyTorch to materialize the full
        # (batch, heads, tgt_len, src_len) weight tensor and skip its
        # fused/memory-efficient attention kernel. Setting this to False
        # produces numerically equivalent output while avoiding that
        # (often multi-hundred-MB) tensor entirely.
        attn_out, _ = self.attn(q, k, v, need_weights=False)
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
        # See note in CrossAttentionBlock.forward: need_weights=False avoids
        # materializing the (batch, heads, latent_size, latent_size)
        # attention-weight tensor for no functional benefit here.
        attn_out, _ = self.attn(h, h, h, need_weights=False)
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

    Memory note: with `use_checkpoint=True` (default), the cross-attention
    block and each self-attention block are wrapped in
    `torch.utils.checkpoint.checkpoint`. During training this recomputes
    each block's activations on the backward pass instead of keeping them
    resident in memory, which is where most of the model's training-time
    RAM usage otherwise goes (the intermediate FF activations alone are
    batch_size x latent_size x (4 x latent_dim) per block). This has no
    effect on the model's outputs or gradients -- checkpointing preserves
    the RNG state across the recomputation so dropout is applied
    identically both times -- it only changes how the same computation is
    scheduled in memory. It is automatically a no-op outside of training
    (e.g. under torch.no_grad()/inference_mode(), where PyTorch keeps no
    activations to begin with).
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
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint

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

        # Only checkpoint during training with grad enabled: under eval /
        # torch.no_grad() there is no backward pass and hence nothing to
        # save memory on, so we skip the (pure) overhead of checkpointing.
        use_ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()

        if use_ckpt:
            z = torch_checkpoint.checkpoint(self.cross_attn, z, x_proj, use_reentrant=False)
        else:
            z = self.cross_attn(z, x_proj)

        for block in self.self_attn_blocks:
            if use_ckpt:
                z = torch_checkpoint.checkpoint(block, z, use_reentrant=False)
            else:
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
    gradient_accumulation_steps: int = 1,
    checkpoint_to_disk: bool = True,
    use_amp: bool = False,
):
    """
    Trains with the KL-divergence loss (Eq. 20 / Section 3.2), AdamW,
    ReduceLROnPlateau, and early stopping -- matching the paper's
    training methodology.

    Memory notes:
      - `gradient_accumulation_steps`: lets you train with a small physical
        batch size (low activation memory per step) while still taking an
        optimizer step every `gradient_accumulation_steps` micro-batches.
        Since KLDivLoss(reduction="batchmean") already averages within each
        micro-batch, dividing the loss by `gradient_accumulation_steps`
        before each backward() makes the accumulated gradient mathematically
        equivalent to training with one large batch of size
        (physical_batch_size * gradient_accumulation_steps) -- i.e. this
        changes peak memory only, not the training objective.
      - `checkpoint_to_disk`: the early-stopping "best model so far" is
        written to a temporary file on disk and reloaded at the end,
        instead of being kept as a second full copy of the model's
        `state_dict()` in RAM for the whole training run.

    Speed note:
      - `use_amp`: enables mixed-precision training via torch.autocast.
        On CUDA this uses float16 with gradient scaling (typically ~1.5-2x
        faster on tensor-core GPUs, negligible accuracy impact for this
        model/loss). On CPU it uses bfloat16 without gradient scaling
        (bfloat16 has the same exponent range as float32, so scaling isn't
        needed) -- CPU speedups from this are hardware-dependent and can be
        small or even negative on CPUs without native bf16 support, so
        benchmark before relying on it there. Disabled (False) by default
        since it is a numerical-precision trade-off, not a pure
        implementation optimization like the other flags in this function.
    """
    model.to(device)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    # GradScaler is only meaningful (and only actually enabled) for the
    # CUDA + float16 combination; with enabled=False it is a transparent
    # pass-through, so it is safe to construct unconditionally.
    scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and device_type == "cuda"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    best_state = None            # used only if checkpoint_to_disk=False
    best_state_path = None       # used only if checkpoint_to_disk=True
    epochs_no_improve = 0

    if checkpoint_to_disk:
        tmp = tempfile.NamedTemporaryFile(prefix="perceiver_best_", suffix=".pt", delete=False)
        best_state_path = tmp.name
        tmp.close()

    try:
        for epoch in tqdm(range(epochs), desc="Epochs"):
            model.train()
            optimizer.zero_grad()
            n_batches = len(train_loader)
            for step, (x_in, _pnoisy, pideal) in enumerate(train_loader):
                x_in, pideal = x_in.to(device), pideal.to(device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    log_pmit = model(x_in)
                    loss = kldiv(log_pmit, pideal.clamp_min(1e-12)) / gradient_accumulation_steps
                scaler.scale(loss).backward()

                is_last_batch = (step == n_batches - 1)
                if (step + 1) % gradient_accumulation_steps == 0 or is_last_batch:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            model.eval()
            val_losses = []
            with torch.inference_mode():
                for x_in, _pnoisy, pideal in val_loader:
                    x_in, pideal = x_in.to(device), pideal.to(device)
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                        log_pmit = model(x_in)
                        val_losses.append(kldiv(log_pmit, pideal.clamp_min(1e-12)).item())
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            scheduler.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                if checkpoint_to_disk:
                    torch.save(model.state_dict(), best_state_path)
                else:
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            # Release this epoch's now-unused temporaries before the next
            # epoch allocates fresh ones.
            del val_losses
            gc.collect()

        if checkpoint_to_disk and best_state_path is not None and best_val < float("inf"):
            model.load_state_dict(torch.load(best_state_path, map_location=device))
        elif best_state is not None:
            model.load_state_dict(best_state)
    finally:
        if checkpoint_to_disk and best_state_path is not None and os.path.exists(best_state_path):
            os.remove(best_state_path)

    return model


def evaluate_l1rc(model: Perceiver, loader: DataLoader, device: str = "cpu", use_amp: bool = False) -> np.ndarray:
    """Runs the model on `loader` and returns the L1 Relative Change per circuit."""
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    all_rc = []
    # inference_mode is a strict superset of no_grad's savings: it also
    # skips PyTorch's autograd view/version-counter bookkeeping, at no cost
    # here since none of these tensors are ever used in a backward pass.
    with torch.inference_mode():
        for x_in, pnoisy, pideal in loader:
            x_in = x_in.to(device)
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                pmit = model(x_in).exp()
            # .numpy() does not support bfloat16/float16 tensors directly;
            # cast back to float32 first. This happens after the
            # compute-heavy forward pass, so it costs nothing meaningful.
            pmit = pmit.float().cpu().numpy()
            rc = l1_relative_change(pmit, pideal.numpy(), pnoisy.numpy())
            all_rc.append(rc)
            del x_in, pmit
    return np.concatenate(all_rc)


# ---------------------------------------------------------------------------
# Figure 6a reproduction
# ---------------------------------------------------------------------------

def reproduce_figure_6a(
    npz_path: str = "pauli_simulated_dataset.npz",
    feature_dim: int = 132,
    epochs: int = 50,
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    device: str = "auto",
    seed: int = 0,
    save_path: str = "figure_6a_perceiver.png",
    use_checkpoint: bool = None,
    use_amp: bool = None,
    checkpoint_to_disk: bool = True,
    lazy_dataset_dir: str = None,
    num_workers: int = 0,
    n_cpu_threads: int = None,
):
    """
    Reproduces the PERCEIVER result of Figure 6a: "Trained on Algiers Pauli
    Simulated - L1 Relative Change" (box plot; paper reports PERCEIVER
    median approx -0.57 with ~91% of circuits improved, Table 14).

    Steps (matching Section 3.2 methodology):
      1. Load the Pauli Simulated (ibm_algiers-style) dataset produced by
         pauli_dataset.build_pauli_simulated_dataset() -- either fully into
         RAM (default, via `npz_path`) or memory-mapped from disk (pass
         `lazy_dataset_dir`, a directory previously populated by
         `save_dataset_npy`).
      2. Split into 50% train / 12.5% val / 37.5% test.
      3. Train a Perceiver ("prediction" model) with KL-divergence loss.
      4. Evaluate the L1 Relative Change (Eq. 21) on the held-out test set
         and render it as a box plot (whiskers at 1st-99th percentile),
         matching the styling of Figure 6a.

    Device handling:
      `device="auto"` (default) picks CUDA if available, else CPU. Any of
      `batch_size`, `gradient_accumulation_steps`, `use_checkpoint`, `use_amp`
      left as None are filled in from `recommended_training_config(device)`
      -- memory-conservative + checkpointed on CPU, larger-batch +
      mixed-precision + uncheckpointed on CUDA (see that function's
      docstring for the reasoning). Pass explicit values to override any of
      these, e.g. to reduce batch_size further if you hit an OOM, or to
      re-enable checkpointing on a more memory-constrained CPU machine.

    `n_cpu_threads`: if set, calls `configure_cpu_threads(n_cpu_threads)`
    before training. Worth setting explicitly to your physical core count
    on CPU-only machines -- some environments leave PyTorch using far fewer
    threads than are actually available (call
    `torch.get_num_threads()` to check first).
    """
    torch.manual_seed(seed)
    device = pick_device(device)

    if n_cpu_threads is not None and device == "cpu":
        actual = configure_cpu_threads(n_cpu_threads)
        print(f"Configured {actual} CPU thread(s) for PyTorch "
              f"(torch.get_num_threads() now reports {torch.get_num_threads()})")

    defaults = recommended_training_config(device)
    if batch_size is None:
        batch_size = defaults["batch_size"]
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = defaults["gradient_accumulation_steps"]
    if use_checkpoint is None:
        use_checkpoint = defaults["use_checkpoint"]
    if use_amp is None:
        use_amp = defaults["use_amp"]

    print(f"Device={device}  batch_size={batch_size}  "
          f"gradient_accumulation_steps={gradient_accumulation_steps}  "
          f"use_checkpoint={use_checkpoint}  use_amp={use_amp}")

    if lazy_dataset_dir is not None:
        full_ds = LazyQEMDataset(lazy_dataset_dir)
        n_outcomes = full_ds.Pideal.shape[-1]
    else:
        data = np.load(npz_path)
        C, B, Pnoisy, Pideal = data["C"], data["B"], data["Pnoisy"], data["Pideal"]
        full_ds = QEMDataset(C, B, Pnoisy, Pideal, subset_fraction=0.1, seed=seed)
        n_outcomes = Pnoisy.shape[-1]

    n = len(full_ds)
    n_train = int(0.50 * n)
    n_val = int(0.125 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    pin_memory = (device == "cuda")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, feature_dim=feature_dim),
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)
    test_loader = make_loader(test_ds, False)

    input_dim = feature_dim + n_outcomes
    model = Perceiver(input_dim=input_dim, n_outcomes=n_outcomes, use_checkpoint=use_checkpoint)

    param_info = count_parameters(model)
    print(f"Perceiver parameter count: {param_info['TOTAL']['params']:,} "
          f"({param_info['TOTAL']['mb_fp32']:.1f} MB fp32)")

    model = train_perceiver(
        model, train_loader, val_loader, epochs=epochs, device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_to_disk=checkpoint_to_disk,
        use_amp=use_amp,
    )
    rc = evaluate_l1rc(model, test_loader, device=device, use_amp=use_amp)

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