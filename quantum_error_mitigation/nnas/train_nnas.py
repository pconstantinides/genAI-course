"""
Trains and evaluates NNAS variants (Original, Dual-State, Physics-Informed)
across four noise conditions (coherent_noise_dataset.CONDITIONS), with
identical optimizer/epochs/batch settings per architecture, plus a
limited-training-data sweep. See coherent_noise_dataset.py / model.py for
the datasets and architectures themselves.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from coherent_noise_dataset import generate_coherent_dataset, CONDITIONS
from model import NNASForQEM, DualStateNNASForQEM, PhysicsInformedNNASForQEM

EPOCHS = 30
LR = 3e-3
N_QUBITS = 4
FIXED_L = 20
N_TRAIN_DEFAULT = 60
N_TEST = 80
PARTIAL_TRAINING_RATE = 0.25
BETA_KL = 1e-3       # Task 7 KL weight for the physics-informed model
VAL_FRAC = 0.15      # held-out fraction for early stopping
PATIENCE = 5         # epochs without validation improvement before stopping
WEIGHT_DECAY = 1e-4   # L2 penalty, alongside gradient clipping
N_SEEDS = 3
BATCH_SIZE = 4        # kept small: larger batches converge to a worse plateau on 'drift'

ARCHITECTURES = {
    "Original NNAS":     dict(cls="single"),
    "Dual-State (full)": dict(cls="dual", use_stochastic=True,  use_coherent=True),
    "Stochastic-only":   dict(cls="dual", use_stochastic=True,  use_coherent=False),
    "Coherent-only":     dict(cls="dual", use_stochastic=False, use_coherent=True),
    "Physics-Informed (generative)": dict(cls="physics"),
}


# ------------------------------------------------------------------------
# Model construction / checkpointing
# ------------------------------------------------------------------------
def build_model(spec_dim: int, arch_name: str, seed: int = 0) -> nn.Module:
    """Seeds torch before constructing the model, so weight init is
    actually controlled by `seed` (must happen before construction, not
    inside the training loop)."""
    torch.manual_seed(seed)
    cfg = ARCHITECTURES[arch_name]
    if cfg["cls"] == "single":
        return NNASForQEM(spec_dim=spec_dim, hidden_dim=32, d=8)
    if cfg["cls"] == "physics":
        return PhysicsInformedNNASForQEM(spec_dim=spec_dim, hidden_dim=32, d=8)
    return DualStateNNASForQEM(spec_dim=spec_dim, hidden_dim=32, d=8,
                                use_stochastic=cfg["use_stochastic"], use_coherent=cfg["use_coherent"])


def save_model(model, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata or {}}, path)
    return path


def load_model(model, path):
    payload = torch.load(Path(path), map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_model_metadata(path):
    path = Path(path)
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    return payload.get("metadata", {}) if isinstance(payload, dict) else {}


# ------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------
def to_tensors(seq, L_max):
    return (
        torch.tensor(seq.spec_features(L_max), dtype=torch.float32).unsqueeze(0),
        torch.tensor(seq.y_noisy, dtype=torch.float32).unsqueeze(0),
        torch.tensor(seq.p_hat, dtype=torch.float32).unsqueeze(0),
        torch.tensor(seq.y_noiseless, dtype=torch.float32).unsqueeze(0),
    )


def _collate_batch(seqs, L_max):
    """Pad variable-length sequences to the batch's max length + build a
    validity mask. Padding only extends the end, and the RNN is causal, so
    it never affects the valid (unmasked) positions."""
    B, max_L = len(seqs), max(s.L for s in seqs)
    spec_dim = seqs[0].spec_features(L_max).shape[-1]
    specs = np.zeros((B, max_L, spec_dim), dtype=np.float32)
    noisy_y = np.zeros((B, max_L), dtype=np.float32)
    p_hat = np.zeros((B, max_L), dtype=np.float32)
    y_true = np.zeros((B, max_L), dtype=np.float32)
    mask = np.zeros((B, max_L), dtype=np.float32)
    for i, s in enumerate(seqs):
        specs[i, :s.L] = s.spec_features(L_max)
        noisy_y[i, :s.L] = s.y_noisy
        p_hat[i, :s.L] = s.p_hat
        y_true[i, :s.L] = s.y_noiseless
        mask[i, :s.L] = 1.0
    return tuple(torch.tensor(a) for a in (specs, noisy_y, p_hat, y_true, mask))


def _masked_loss(out, y_true, mask, beta_kl):
    """Mitigation MSE, plus the Task 7 KL term when the model returns one
    (out = (y_em, r, kl_seq) for the physics-informed model vs. (y_em, r)
    otherwise)."""
    y_em = out[0]
    loss = ((y_em - y_true) ** 2 * mask).sum() / mask.sum()
    if len(out) == 3:
        loss = loss + beta_kl * (out[2] * mask).sum() / mask.sum()
    return loss


def _run_epoch(model, seqs, L_max, batch_size, beta_kl, opt=None, rng=None):
    """One pass over `seqs`. Trains (opt given) or just evaluates (opt=None)."""
    model.train(opt is not None)
    order = rng.permutation(len(seqs)) if rng is not None else np.arange(len(seqs))
    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for start in range(0, len(seqs), batch_size):
            batch = [seqs[i] for i in order[start:start + batch_size]]
            specs, noisy_y, p_hat, y_true, mask = _collate_batch(batch, L_max)
            out = model(specs, noisy_y, p_hat)
            loss = _masked_loss(out, y_true, mask, beta_kl)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
            total_loss += loss.item() * len(batch)
            total_n += len(batch)
    return total_loss / total_n


def train_model(model, train_seqs, L_max, epochs=EPOCHS, lr=LR, seed=0, batch_size=BATCH_SIZE,
                 beta_kl=BETA_KL, val_frac=VAL_FRAC, patience=PATIENCE, weight_decay=WEIGHT_DECAY):
    """
    Batched training with a held-out validation split for early stopping
    (stops after `patience` epochs without improvement, restores the
    best-validation weights) and L2 weight decay as a second, complementary
    overfitting guard alongside gradient clipping.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_val = max(1, round(len(train_seqs) * val_frac)) if len(train_seqs) > 3 else 0
    perm = rng.permutation(len(train_seqs))
    val_seqs = [train_seqs[i] for i in perm[:n_val]] if n_val else train_seqs
    fit_seqs = [train_seqs[i] for i in perm[n_val:]] if n_val else train_seqs

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, stale = float("inf"), None, 0
    final_loss = None
    for _ in range(epochs):
        final_loss = _run_epoch(model, fit_seqs, L_max, batch_size, beta_kl, opt=opt, rng=rng)
        val_loss = _run_epoch(model, val_seqs, L_max, batch_size, beta_kl)

        if val_loss < best_val - 1e-6:
            best_val, best_state, stale = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return final_loss


# ------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------
@dataclass
class EvalMetrics:
    mae: float
    mse: float
    mae_noisy: float
    mse_noisy: float
    rel_improvement_pct: float
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
            y_em = model(specs, noisy_y, p_hat)[0].squeeze(0).numpy()

            diff_noisy = seq.y_noisy - seq.y_noiseless
            diff_nnas = y_em - seq.y_noiseless
            err_noisy[i, :seq.L] = np.abs(diff_noisy)
            err_nnas[i, :seq.L] = np.abs(diff_nnas)
            sqerr_noisy[i, :seq.L] = diff_noisy ** 2
            sqerr_nnas[i, :seq.L] = diff_nnas ** 2

    mae_noisy, mae_nnas = float(np.nanmean(err_noisy)), float(np.nanmean(err_nnas))
    deep = int(0.75 * L_max)
    deep_noisy, deep_nnas = float(np.nanmean(err_noisy[:, deep:])), float(np.nanmean(err_nnas[:, deep:]))

    return EvalMetrics(
        mae=mae_nnas, mse=float(np.nanmean(sqerr_nnas)),
        mae_noisy=mae_noisy, mse_noisy=float(np.nanmean(sqerr_noisy)),
        rel_improvement_pct=100 * (1 - mae_nnas / mae_noisy) if mae_noisy > 0 else float("nan"),
        deep_quartile_mae=deep_nnas, deep_quartile_mae_noisy=deep_noisy,
        deep_quartile_rel_improvement_pct=100 * (1 - deep_nnas / deep_noisy) if deep_noisy > 0 else float("nan"),
        per_layer_mae=np.nanmean(err_nnas, axis=0),
        per_layer_mae_noisy=np.nanmean(err_noisy, axis=0),
    )


# ------------------------------------------------------------------------
# Task 3+4: comparison across conditions, multi-seed
# ------------------------------------------------------------------------
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

    train_seqs = generate_coherent_dataset(condition, n_sequences=n_train, n_qubits=N_QUBITS, fixed_L=FIXED_L,
                                            is_train=True, partial_training_rate=PARTIAL_TRAINING_RATE, seed=1000)
    test_seqs = generate_coherent_dataset(condition, n_sequences=N_TEST, n_qubits=N_QUBITS, fixed_L=FIXED_L,
                                           is_train=False, seed=2000)
    spec_dim = train_seqs[0].spec_features(FIXED_L).shape[-1]

    results = {}
    for arch_name in ARCHITECTURES:
        t0 = time.time()
        rel, deep = [], []
        for seed in range(n_seeds):
            model = build_model(spec_dim, arch_name, seed=seed)
            train_model(model, train_seqs, FIXED_L, seed=seed, batch_size=BATCH_SIZE)
            m = evaluate_model(model, test_seqs, FIXED_L)
            rel.append(m.rel_improvement_pct)
            deep.append(m.deep_quartile_rel_improvement_pct)

        results[arch_name] = SeedAggregate(
            mean_rel_improvement=float(np.mean(rel)), std_rel_improvement=float(np.std(rel)),
            mean_deep_rel_improvement=float(np.mean(deep)), std_deep_rel_improvement=float(np.std(deep)),
            per_seed=rel,
        )
        if verbose:
            print(f"  [{arch_name:<18}] {np.mean(rel):+6.1f}% +/- {np.std(rel):4.1f}%  "
                  f"(deep {np.mean(deep):+6.1f}% +/- {np.std(deep):4.1f}%)  [{time.time()-t0:.1f}s]")
    return results


def print_summary_table(all_results: dict):
    header = f"{'Architecture':<20}" + "".join(f"{c:>22}" for c in CONDITIONS)

    def _print_block(title, field_mean, field_std):
        print(f"\n{'='*len(header)}\n{title}\n{'='*len(header)}\n{header}\n{'-'*len(header)}")
        for arch_name in ARCHITECTURES:
            row = f"{arch_name:<20}"
            for condition in CONDITIONS:
                agg = all_results[condition][arch_name]
                row += f"{getattr(agg, field_mean):>+8.1f}% +/-{getattr(agg, field_std):>5.1f}%"
            print(row)

    _print_block(f"Mean +/- std relative MAE improvement over noisy baseline (%), {N_SEEDS} seeds",
                 "mean_rel_improvement", "std_rel_improvement")
    _print_block("Deepest-quartile-of-layers relative MAE improvement (%)",
                 "mean_deep_rel_improvement", "std_deep_rel_improvement")


# ------------------------------------------------------------------------
# Task 4: generalization with limited training data
# ------------------------------------------------------------------------
def run_limited_data_sweep(condition: str, train_sizes=(20, 50, 100), n_seeds=N_SEEDS):
    print(f"\n{'#'*78}\n# Limited-data sweep: {condition.upper()}\n{'#'*78}")
    test_seqs = generate_coherent_dataset(condition, n_sequences=N_TEST, n_qubits=N_QUBITS, fixed_L=FIXED_L,
                                           is_train=False, seed=3000)
    rows = {}
    for n_train in train_sizes:
        train_seqs = generate_coherent_dataset(condition, n_sequences=n_train, n_qubits=N_QUBITS, fixed_L=FIXED_L,
                                                 is_train=True, partial_training_rate=PARTIAL_TRAINING_RATE, seed=4000)
        spec_dim = train_seqs[0].spec_features(FIXED_L).shape[-1]
        for arch_name in ("Original NNAS", "Dual-State (full)"):
            rel = []
            for seed in range(n_seeds):
                model = build_model(spec_dim, arch_name, seed=seed)
                train_model(model, train_seqs, FIXED_L, seed=seed, batch_size=BATCH_SIZE)
                rel.append(evaluate_model(model, test_seqs, FIXED_L).rel_improvement_pct)
            rows.setdefault(arch_name, {})[n_train] = (float(np.mean(rel)), float(np.std(rel)))

    header = f"{'n_train':<20}" + "".join(f"{n:>16}" for n in train_sizes)
    print(header, "-" * len(header), sep="\n")
    for arch_name in ("Original NNAS", "Dual-State (full)"):
        row = f"{arch_name:<20}"
        for n_train in train_sizes:
            m, s = rows[arch_name][n_train]
            row += f"{m:>+9.1f}%+/-{s:<4.1f}"
        print(row)
    return rows


if __name__ == "__main__":
    all_results = {c: run_condition(c, n_train=N_TRAIN_DEFAULT) for c in CONDITIONS}
    print_summary_table(all_results)
    for condition in ("coherent", "drift"):
        run_limited_data_sweep(condition, train_sizes=(20, 50, 100))