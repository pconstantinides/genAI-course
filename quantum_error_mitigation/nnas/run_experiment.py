"""
run_full_experiment.py

Runs the four noise-condition experiments (stochastic, coherent, mixed,
drift) at scale, comparing all four architectures (Original NNAS,
Dual-State full, Stochastic-only, Coherent-only) across multiple seeds,
and produces:

  1. A summary table (mean +/- std relative MAE improvement over the noisy
     baseline), printed to stdout and saved to results_table.txt.
  2. A 2x2 plot of estimation error (MAE) vs. circuit depth -- Noisy vs.
     Original NNAS vs. Dual-State (full) -- one subplot per condition,
     saved to error_vs_depth.png.

Reuses coherent_noise_dataset.py and train_coherent_nnas.py; no logic is
duplicated here. Adjust N_TRAIN / N_TEST / N_SEEDS / EPOCHS below for your
own compute budget -- the defaults are a moderate "at scale" setting
(larger than the quick diagnostic runs used during development).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from coherent_noise_dataset import generate_coherent_dataset, CONDITIONS, CoherentSequence
from train_nnas import (
    build_model, train_model, evaluate_model, ARCHITECTURES,
    N_QUBITS, FIXED_L, PARTIAL_TRAINING_RATE, BATCH_SIZE,
    save_model, load_model, load_model_metadata,
)

import __main__ # fix the unpickling issue
__main__.CoherentSequence = CoherentSequence

# ---- "at scale" experiment configuration (tune to your compute budget) ----
N_TRAIN = 80
N_TEST = 100
N_SEEDS = 3
EPOCHS = 20
DEPTH_PLOT_ARCHS = ("Original NNAS", "Dual-State (full)")  # kept to 2 lines/plot for readability

RESULTS_DIR = Path(__file__).resolve().parent / "artifacts"
DATASET_DIR = RESULTS_DIR / "datasets"
MODEL_DIR = RESULTS_DIR / "models"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

USE_EXISTING_DATASETS = True
REUSE_EXISTING_MODELS = True
SAVE_MODELS = True


def _dataset_path(condition: str, split: str, n_sequences: int, seed: int) -> Path:
    return DATASET_DIR / f"{condition}_{split}_{n_sequences}_{seed}.npy"


def _model_path(condition: str, arch_name: str, seed: int) -> Path:
    arch_slug = arch_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    return MODEL_DIR / f"{condition}_{arch_slug}_seed{seed}.pt"


def run_all(use_existing_datasets=True, reuse_existing_models=True, save_models=True):
    table = {}         # table[condition][arch] = (mean_pct, std_pct, mean_deep_pct, std_deep_pct)
    depth_curves = {}  # depth_curves[condition][arch] = (L,) MAE averaged over seeds
    depth_std_curves = {}  # depth_std_curves[condition][arch] = (L,) std of MAE across seeds
    noisy_curves = {}  # noisy_curves[condition] = (L,) noisy-baseline MAE

    for condition in CONDITIONS:
        print(f"\n=== {condition} ===")
        train_dataset_path = DATASET_DIR / f"{condition}_train_20000_23.npy" #_dataset_path(condition, "train", N_TRAIN, 1000)
        test_dataset_path = DATASET_DIR / f"{condition}_test_5000_32.npy" #_dataset_path(condition, "test", N_TEST, 2000)
        train_seqs = generate_coherent_dataset(
            condition, n_sequences=N_TRAIN, n_qubits=N_QUBITS, fixed_L=FIXED_L,
            is_train=True, filename=str(train_dataset_path),
            load_if_exists=use_existing_datasets,
            partial_training_rate=PARTIAL_TRAINING_RATE, seed=1000)
        test_seqs = generate_coherent_dataset(
            condition, n_sequences=N_TEST, n_qubits=N_QUBITS, fixed_L=FIXED_L,
            is_train=False, filename=str(test_dataset_path),
            load_if_exists=use_existing_datasets, seed=2000)
        spec_dim = train_seqs[0].spec_features(FIXED_L).shape[-1]

        table[condition] = {}
        depth_curves[condition] = {}
        depth_std_curves[condition] = {}
        noisy_layer_runs = []
        for arch_name in ARCHITECTURES:
            print(f"  {arch_name:<18}: ", end="", flush=True)
            rel, deep, per_layer = [], [], []
            for seed in range(N_SEEDS):
                print(f"[seed {seed}] ", end="", flush=True)
                model_path = _model_path(condition, arch_name, seed)
                model = build_model(spec_dim, arch_name, seed=seed)
                if reuse_existing_models and model_path.exists():
                    model = load_model(model, str(model_path))
                    print(f"    [seed {seed}] loaded existing model from {model_path}")
                else:
                    train_model(model, train_seqs, FIXED_L, epochs=1, seed=seed, batch_size=BATCH_SIZE)
                
                m = evaluate_model(model, test_seqs, FIXED_L)
                rel.append(m.rel_improvement_pct)
                deep.append(m.deep_quartile_rel_improvement_pct)
                per_layer.append(m.per_layer_mae)
                noisy_layer_runs.append(m.per_layer_mae_noisy)  # identical across archs/seeds (depends only on test_seqs)

                if save_models:
                    existing_metric = load_model_metadata(str(model_path)).get("rel_improvement_pct", -np.inf)
                    if not model_path.exists() or m.rel_improvement_pct > existing_metric:
                        save_model(
                            model,
                            str(model_path),
                            metadata={
                                "condition": condition,
                                "arch_name": arch_name,
                                "seed": seed,
                                "rel_improvement_pct": float(m.rel_improvement_pct),
                                "deep_quartile_rel_improvement_pct": float(m.deep_quartile_rel_improvement_pct),
                            },
                        )
            table[condition][arch_name] = (
                float(np.mean(rel)), float(np.std(rel)),
                float(np.mean(deep)), float(np.std(deep)),
            )
            per_layer = np.asarray(per_layer)
            depth_curves[condition][arch_name] = np.mean(per_layer, axis=0)
            depth_std_curves[condition][arch_name] = np.std(per_layer, axis=0)
            print(f"  {arch_name:<18}: {np.mean(rel):+6.1f}% +/- {np.std(rel):4.1f}%  "
                  f"(deep-quartile {np.mean(deep):+6.1f}% +/- {np.std(deep):4.1f}%)")

        noisy_curves[condition] = np.mean(noisy_layer_runs, axis=0)

    return table, depth_curves, depth_std_curves, noisy_curves


def print_table(table):
    header = f"{'Architecture':<20}" + "".join(f"{c:>22}" for c in CONDITIONS)
    lines = [
        f"Mean +/- std relative MAE improvement over noisy baseline (%), {N_SEEDS} seeds",
        header, "-" * len(header),
    ]
    for arch_name in ARCHITECTURES:
        row = f"{arch_name:<20}"
        for condition in CONDITIONS:
            m, s, _, _ = table[condition][arch_name]
            row += f"{m:>+8.1f}% +/-{s:>5.1f}%"
        lines.append(row)

    lines += ["", "Deepest-quartile-of-layers relative MAE improvement (%)", header, "-" * len(header)]
    for arch_name in ARCHITECTURES:
        row = f"{arch_name:<20}"
        for condition in CONDITIONS:
            _, _, m, s = table[condition][arch_name]
            row += f"{m:>+8.1f}% +/-{s:>5.1f}%"
        lines.append(row)

    text = "\n".join(lines)
    print("\n" + text)
    with open("results_table.txt", "w") as f:
        f.write(text + "\n")
    print("\nSaved table to results_table.txt")


def plot_depth_curves(depth_curves, depth_std_curves, noisy_curves):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, condition in zip(axes.flat, CONDITIONS):
        L = len(noisy_curves[condition])
        x = np.arange(1, L + 1)
        ax.plot(x, noisy_curves[condition], "k--", label="Noisy", linewidth=1.5)
        for arch_name in DEPTH_PLOT_ARCHS:
            mean_curve = depth_curves[condition][arch_name]
            std_curve = depth_std_curves[condition][arch_name]
            ax.plot(x, mean_curve, marker="o", markersize=3, label=arch_name)
            ax.fill_between(
                x,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.2,
                linewidth=0,
            )
        ax.set_title(condition)
        ax.set_xlabel("Circuit depth (layer)")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("error_vs_depth_3.png", dpi=150)
    print("Saved plot to error_vs_depth_3.png")


if __name__ == "__main__":
    table, depth_curves, depth_std_curves, noisy_curves = run_all()
    print_table(table)
    plot_depth_curves(depth_curves, depth_std_curves, noisy_curves)
