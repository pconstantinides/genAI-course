"""
Research Plan: "Dual-State NNAS with Lie-Algebra Inspired Coherent Error
Modeling" -- Tasks 3 (training), 4 (evaluation), 5 (ablation study).

Compares four architectures --
  - Original NNAS            (nnas_model.NNASForQEM)
  - Dual-State NNAS           (DualStateNNASForQEM, both branches)
  - Stochastic-branch only    (DualStateNNASForQEM, use_coherent=False)
  - Coherent-branch only      (DualStateNNASForQEM, use_stochastic=False)
across four noise conditions (coherent_noise_dataset.CONDITIONS):
  stochastic, coherent, mixed, drift

under IDENTICAL optimizer / learning rate / epochs / batch (=1 sequence,
same as the rest of this codebase) settings, per Task 3's requirement.
Also runs a small limited-training-data sweep (Task 4's "generalization
with limited training data") for the two architectures of primary
interest (Original vs. full Dual-State).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from coherent_noise_dataset import generate_coherent_dataset, CONDITIONS
from model import NNASForQEM, DualStateNNASForQEM

EPOCHS = 120
LR = 3e-3
N_QUBITS = 4
FIXED_L = 16
N_TRAIN_DEFAULT = 60
N_TEST = 80
PARTIAL_TRAINING_RATE = 0.25

ARCHITECTURES = {
    "Original NNAS":     dict(cls="single"),
    "Dual-State (full)": dict(cls="dual", use_stochastic=True,  use_coherent=True),
    "Stochastic-only":   dict(cls="dual", use_stochastic=True,  use_coherent=False),
    "Coherent-only":     dict(cls="dual", use_stochastic=False, use_coherent=True),
}


# ----------------------------------------------------------------------
# Shared tensor conversion (same convention as train_nnas.py)
# ----------------------------------------------------------------------
def to_tensors(seq, L_max):
    specs = torch.tensor(seq.spec_features(L_max), dtype=torch.float32).unsqueeze(0)
    noisy_y = torch.tensor(seq.y_noisy, dtype=torch.float32).unsqueeze(0)
    p_hat = torch.tensor(seq.p_hat, dtype=torch.float32).unsqueeze(0)
    y_true = torch.tensor(seq.y_noiseless, dtype=torch.float32).unsqueeze(0)
    return specs, noisy_y, p_hat, y_true


def save_model(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_model(model, path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_metadata(path):
    path = Path(path)
    if not path.exists():
        return {}

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "metadata" in payload:
        return payload["metadata"]
    return {}


def build_model(spec_dim: int, arch_name: str, seed: int = 0) -> nn.Module:
    """IMPORTANT: seeds torch's global RNG BEFORE constructing the model,
    so weight initialization is actually controlled by `seed`. Previously
    (bug, found via the drift-plateau investigation) `torch.manual_seed`
    was only called inside the training loop, i.e. AFTER the model's
    weights were already initialized from whatever the global RNG state
    happened to be -- which depends on how many other models were built
    earlier in the same process/loop. That made architecture comparisons
    silently confounded by uncontrolled initialization noise, on top of
    whatever genuine architectural effect existed."""
    torch.manual_seed(seed)
    cfg = ARCHITECTURES[arch_name]
    if cfg["cls"] == "single":
        return NNASForQEM(spec_dim=spec_dim, hidden_dim=32, d=8, use_noisy_results=True)
    return DualStateNNASForQEM(
        spec_dim=spec_dim, hidden_dim=32, d=8, use_noisy_results=True,
        use_stochastic=cfg["use_stochastic"], use_coherent=cfg["use_coherent"],
    )


def train_model_unbatched(model, train_seqs, L_max, epochs=EPOCHS, lr=LR, seed=0):
    """Original, unbatched (one sequence per gradient step) training loop.
    Kept for reference/correctness-verification against the batched path
    below; NOT used by the main sweep (too slow at this experiment's
    scale -- see the profiling note in train_model_batched)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    final_loss = None
    for epoch in range(epochs):
        perm = rng.permutation(len(train_seqs))
        total_loss = 0.0
        for idx in perm:
            seq = train_seqs[idx]
            specs, noisy_y, p_hat, y_true = to_tensors(seq, L_max)

            opt.zero_grad()
            y_em, _ = model(specs, noisy_y, p_hat)
            loss = loss_fn(y_em, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        final_loss = total_loss / len(train_seqs)
    return final_loss


def _collate_batch(seqs, L_max):
    """Pad a list of variable-length sequences to their batch's max length
    and build a validity mask. Padding only ever extends the END of a
    sequence, and the RNN core is causal (H_l depends only on X_1..X_l), so
    padding cannot leak into the loss-relevant (valid) positions -- it only
    ever adds extra, masked-out tail computation."""
    B = len(seqs)
    max_L = max(s.L for s in seqs)
    spec_dim = seqs[0].spec_features(L_max).shape[-1]

    specs = np.zeros((B, max_L, spec_dim), dtype=np.float32)
    noisy_y = np.zeros((B, max_L), dtype=np.float32)
    p_hat = np.zeros((B, max_L), dtype=np.float32)
    y_true = np.zeros((B, max_L), dtype=np.float32)
    mask = np.zeros((B, max_L), dtype=np.float32)

    for i, s in enumerate(seqs):
        L = s.L
        specs[i, :L] = s.spec_features(L_max)
        noisy_y[i, :L] = s.y_noisy
        p_hat[i, :L] = s.p_hat
        y_true[i, :L] = s.y_noiseless
        mask[i, :L] = 1.0

    return (torch.tensor(specs), torch.tensor(noisy_y), torch.tensor(p_hat),
            torch.tensor(y_true), torch.tensor(mask))


def train_model(model, train_seqs, L_max, epochs=EPOCHS, lr=LR, seed=0, batch_size=20):
    """
    Batched training loop (padding + masked MSE loss). Same optimizer/lr/
    epochs semantics as the unbatched version -- the only change is how
    many sequences are processed per gradient step.

    Why this matters: profiling the unbatched loop (cProfile, 10 real
    epochs on the 'drift' dataset) showed loss.backward() alone consuming
    42% of total wall time, dominated by PyTorch's per-call autograd graph
    construction/teardown overhead (~2.1ms fixed cost per call, vs. ~0.76ms
    of actual per-timestep RNN compute) -- overhead that's *not* amortized
    when every gradient step processes exactly one (typically short, mean
    L~9 out of 16 under the hard-regime truncation scheme) sequence.
    Batching amortizes that fixed cost across `batch_size` sequences per
    step instead of one, since NNASCore/DualStateNNASCore already operate
    natively on (batch, L, feature_dim) tensors -- nothing in the model
    needed to change.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    final_loss = None
    for epoch in range(epochs):
        perm = rng.permutation(len(train_seqs))
        total_loss, total_count = 0.0, 0
        for start in range(0, len(train_seqs), batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_seqs = [train_seqs[i] for i in batch_idx]
            specs, noisy_y, p_hat, y_true, mask = _collate_batch(batch_seqs, L_max)

            opt.zero_grad()
            y_em, _ = model(specs, noisy_y, p_hat)
            sq_err = (y_em - y_true) ** 2 * mask
            loss = sq_err.sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * len(batch_seqs)
            total_count += len(batch_seqs)
        final_loss = total_loss / total_count
    return final_loss


@dataclass
class EvalMetrics:
    mae: float
    mse: float
    mae_noisy: float
    mse_noisy: float
    rel_improvement_pct: float          # 100*(1 - mae/mae_noisy)
    deep_quartile_mae: float
    deep_quartile_mae_noisy: float
    deep_quartile_rel_improvement_pct: float
    per_layer_mae: np.ndarray = field(repr=False)
    per_layer_mae_noisy: np.ndarray = field(repr=False)


def evaluate_model(model, test_seqs, L_max) -> EvalMetrics:
    model.eval()
    n = len(test_seqs)
    err_noisy = np.full((n, L_max), np.nan)
    err_nnas = np.full((n, L_max), np.nan)
    sqerr_noisy = np.full((n, L_max), np.nan)
    sqerr_nnas = np.full((n, L_max), np.nan)

    with torch.no_grad():
        for i, seq in enumerate(test_seqs):
            specs, noisy_y, p_hat, y_true = to_tensors(seq, L_max)
            y_em, _ = model(specs, noisy_y, p_hat)
            y_em_np = y_em.squeeze(0).numpy()

            diff_noisy = seq.y_noisy - seq.y_noiseless
            diff_nnas = y_em_np - seq.y_noiseless

            err_noisy[i, :seq.L] = np.abs(diff_noisy)
            err_nnas[i, :seq.L] = np.abs(diff_nnas)
            sqerr_noisy[i, :seq.L] = diff_noisy ** 2
            sqerr_nnas[i, :seq.L] = diff_nnas ** 2

    mae_noisy = float(np.nanmean(err_noisy))
    mae_nnas = float(np.nanmean(err_nnas))
    mse_noisy = float(np.nanmean(sqerr_noisy))
    mse_nnas = float(np.nanmean(sqerr_nnas))

    deep_start = int(0.75 * L_max)
    deep_noisy = float(np.nanmean(err_noisy[:, deep_start:]))
    deep_nnas = float(np.nanmean(err_nnas[:, deep_start:]))

    return EvalMetrics(
        mae=mae_nnas, mse=mse_nnas,
        mae_noisy=mae_noisy, mse_noisy=mse_noisy,
        rel_improvement_pct=100 * (1 - mae_nnas / mae_noisy) if mae_noisy > 0 else float("nan"),
        deep_quartile_mae=deep_nnas, deep_quartile_mae_noisy=deep_noisy,
        deep_quartile_rel_improvement_pct=100 * (1 - deep_nnas / deep_noisy) if deep_noisy > 0 else float("nan"),
        per_layer_mae=np.nanmean(err_nnas, axis=0),
        per_layer_mae_noisy=np.nanmean(err_noisy, axis=0),
    )


# ----------------------------------------------------------------------
# Task 3+4: main comparison across the four conditions
#
# Multi-seed by construction: build_model's seed now genuinely controls
# weight init (see the fix above), and run_condition trains N_SEEDS
# independent models per architecture (same data, different init +
# batch order) and reports mean +/- std. This replaces the earlier
# single-seed point estimates, which -- given the small (single-digit
# percent) effect sizes in this study -- were not distinguishable from
# initialization noise on their own.
#
# batch_size=4 (rather than the faster but dynamics-changing batch_size=20)
# is used here: the batch-size sweep on 'drift' showed larger batches
# converge to a measurably worse plateau on this noise-oscillatory loss
# landscape, so this keeps optimization dynamics close to the original
# per-sequence SGD while still giving a large (~3x) speedup from batching.
# ----------------------------------------------------------------------
N_SEEDS = 3
BATCH_SIZE = 4


@dataclass
class SeedAggregate:
    mean_rel_improvement: float
    std_rel_improvement: float
    mean_deep_rel_improvement: float
    std_deep_rel_improvement: float
    per_seed: list = field(repr=False)


def run_condition(condition: str, n_train=N_TRAIN_DEFAULT, n_seeds=N_SEEDS, verbose=True):
    if verbose:
        print(f"\n{'#'*78}\n# Condition: {condition.upper()}\n{'#'*78}")

    # Data is fixed across seeds within a condition (only model init +
    # minibatch order vary across seeds) -- isolates architecture/init
    # variance from dataset-sampling variance.
    train_seqs = generate_coherent_dataset(
        condition, n_sequences=n_train, n_qubits=N_QUBITS, fixed_L=FIXED_L,
        is_train=True, partial_training_rate=PARTIAL_TRAINING_RATE, seed=1000,
    )
    test_seqs = generate_coherent_dataset(
        condition, n_sequences=N_TEST, n_qubits=N_QUBITS, fixed_L=FIXED_L,
        is_train=False, seed=2000,
    )
    spec_dim = train_seqs[0].spec_features(FIXED_L).shape[-1]

    results = {}
    for arch_name in ARCHITECTURES:
        t0 = time.time()
        rel_list, deep_list = [], []
        for seed in range(n_seeds):
            model = build_model(spec_dim, arch_name, seed=seed)
            train_model(model, train_seqs, FIXED_L, seed=seed, batch_size=BATCH_SIZE)
            metrics = evaluate_model(model, test_seqs, FIXED_L)
            rel_list.append(metrics.rel_improvement_pct)
            deep_list.append(metrics.deep_quartile_rel_improvement_pct)

        agg = SeedAggregate(
            mean_rel_improvement=float(np.mean(rel_list)), std_rel_improvement=float(np.std(rel_list)),
            mean_deep_rel_improvement=float(np.mean(deep_list)), std_deep_rel_improvement=float(np.std(deep_list)),
            per_seed=rel_list,
        )
        results[arch_name] = agg
        if verbose:
            print(f"  [{arch_name:<18}] MAE improvement = {agg.mean_rel_improvement:+6.1f}% "
                  f"+/- {agg.std_rel_improvement:4.1f}%  (deep-quartile {agg.mean_deep_rel_improvement:+6.1f}% "
                  f"+/- {agg.std_deep_rel_improvement:4.1f}%)  per-seed={[round(x,1) for x in rel_list]}  "
                  f"[{time.time()-t0:.1f}s]")
    return results


def print_summary_table(all_results: dict):
    print("\n" + "=" * 110)
    print("SUMMARY: mean +/- std relative MAE improvement over noisy baseline (%), across "
          f"{N_SEEDS} seeds, by condition x architecture")
    print("=" * 110)
    header = f"{'Architecture':<20}" + "".join(f"{c:>22}" for c in CONDITIONS)
    print(header)
    print("-" * len(header))
    for arch_name in ARCHITECTURES:
        row = f"{arch_name:<20}"
        for condition in CONDITIONS:
            agg = all_results[condition][arch_name]
            row += f"{agg.mean_rel_improvement:>+8.1f}% +/-{agg.std_rel_improvement:>5.1f}%"
        print(row)

    print("\n" + "=" * 110)
    print("Deepest-quartile-of-layers relative MAE improvement over noisy baseline (%)")
    print("=" * 110)
    print(header)
    print("-" * len(header))
    for arch_name in ARCHITECTURES:
        row = f"{arch_name:<20}"
        for condition in CONDITIONS:
            agg = all_results[condition][arch_name]
            row += f"{agg.mean_deep_rel_improvement:>+8.1f}% +/-{agg.std_deep_rel_improvement:>5.1f}%"
        print(row)
    print("=" * 110)


# ----------------------------------------------------------------------
# Task 4: generalization with limited training data
# ----------------------------------------------------------------------
def run_limited_data_sweep(condition: str, train_sizes=(20, 50, 100), n_seeds=N_SEEDS):
    print(f"\n{'#'*78}\n# Limited-data sweep: {condition.upper()}\n{'#'*78}")
    test_seqs = generate_coherent_dataset(
        condition, n_sequences=N_TEST, n_qubits=N_QUBITS, fixed_L=FIXED_L,
        is_train=False, seed=3000,
    )
    rows = {}
    for n_train in train_sizes:
        train_seqs = generate_coherent_dataset(
            condition, n_sequences=n_train, n_qubits=N_QUBITS, fixed_L=FIXED_L,
            is_train=True, partial_training_rate=PARTIAL_TRAINING_RATE, seed=4000,
        )
        spec_dim = train_seqs[0].spec_features(FIXED_L).shape[-1]
        for arch_name in ("Original NNAS", "Dual-State (full)"):
            rel_list = []
            for seed in range(n_seeds):
                model = build_model(spec_dim, arch_name, seed=seed)
                train_model(model, train_seqs, FIXED_L, seed=seed, batch_size=BATCH_SIZE)
                metrics = evaluate_model(model, test_seqs, FIXED_L)
                rel_list.append(metrics.rel_improvement_pct)
            rows.setdefault(arch_name, {})[n_train] = (float(np.mean(rel_list)), float(np.std(rel_list)))

    header = f"{'n_train':<20}" + "".join(f"{n:>16}" for n in train_sizes)
    print(header)
    print("-" * len(header))
    for arch_name in ("Original NNAS", "Dual-State (full)"):
        row = f"{arch_name:<20}"
        for n_train in train_sizes:
            m, s = rows[arch_name][n_train]
            row += f"{m:>+9.1f}%+/-{s:<4.1f}"
        print(row)
    return rows


if __name__ == "__main__":
    all_results = {}
    for condition in CONDITIONS:
        all_results[condition] = run_condition(condition, n_train=N_TRAIN_DEFAULT)

    print_summary_table(all_results)

    for condition in ("coherent", "drift"):
        run_limited_data_sweep(condition, train_sizes=(20, 50, 100))