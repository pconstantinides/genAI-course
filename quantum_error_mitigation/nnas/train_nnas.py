"""
Unified NNAS training/evaluation script, covering both supported circuit
families through ONE shared training loop:

  --task trotter          1D transverse-field Ising Trotter circuits
                           (qem_dataset.py), T1/T2-derived Pauli noise.
  --task real_amplitudes   RealAmplitudes variational ansatz
                           (real_amplitudes_dataset.py), depolarizing noise,
                           generated via (or loaded from) the Qiskit-based
                           generator extended from the user's own dataset
                           script.

Both circuit families expose the same minimal interface on their sequence
objects -- `.L`, `.y_noiseless`, `.y_noisy`, `.p_hat`, `.spec_features(L_max)`
(qem_dataset.Sequence / real_amplitudes_dataset.RASequence) -- so the
training loop below (model construction, optimization, evaluation) doesn't
need to know or care which circuit family produced them. Only the dataset
*construction* step is task-specific.

What we check (the "easy to verify" claim, same as before):
    MAE(NNAS-mitigated) should be substantially lower than MAE(Noisy),
    with the improvement generally growing at deeper layers -- Fig. 2 a-c
    of the paper for Trotter circuits; the analogous per-layer breakdown
    for RealAmplitudes circuits.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from qem_dataset import (
    generate_qaoa_dataset,
    sample_max_length_train_generic,
    truncate_sequence,
)
from real_amplitudes_dataset import (
    generate_real_amplitudes_dataset,
    load_layerwise_dataset,
)
from model import NNASForQEM

EPOCHS = 150
LR = 3e-3


# ----------------------------------------------------------------------
# Task-specific dataset construction. Each branch returns
# (train_seqs, test_seqs, L_max, label) -- everything downstream is shared.
# ----------------------------------------------------------------------
def build_trotter_dataset(args):
    n_qubits, t1_us, L_max = 6, 23.235, 20
    print(f"[trotter] Generating training set ({args.n_train} sequences, "
          f"partial_training_rate={args.partial_training_rate}) ...")
    train_seqs = generate_qaoa_dataset(
        n_sequences=args.n_train, n_qubits=n_qubits, noise_t1_us=t1_us,
        partial_training_rate=args.partial_training_rate, is_train=True,
        fixed_L=L_max, seed=42,
    )
    print(f"[trotter] Generating test set ({args.n_test} sequences, full length {L_max}) ...")
    test_seqs = generate_qaoa_dataset(
        n_sequences=args.n_test, n_qubits=n_qubits, noise_t1_us=t1_us,
        is_train=False, fixed_L=L_max, seed=123,
    )
    return train_seqs, test_seqs, L_max, "Trotter step"


def build_real_amplitudes_dataset(args):
    L_max = args.n_layers
    rng = np.random.default_rng(7)

    if args.data is not None:
        print(f"[real_amplitudes] Loading dataset from {args.data} "
              f"(n_qubits={args.n_qubits}, n_layers={L_max}) ...")
        all_seqs = load_layerwise_dataset(args.data, n_qubits=args.n_qubits, n_layers=L_max)
        perm = rng.permutation(len(all_seqs))
        n_train = min(args.n_train, len(all_seqs) // 2)
        train_full = [all_seqs[i] for i in perm[:n_train]]
        test_seqs = [all_seqs[i] for i in perm[n_train:n_train + args.n_test]]
    else:
        print(f"[real_amplitudes] No --data given: generating synthetic data "
              f"({args.n_train} train / {args.n_test} test, "
              f"n_qubits={args.n_qubits}, n_layers={L_max}) ...")
        train_full = generate_real_amplitudes_dataset(
            n_sequences=args.n_train, n_qubits=args.n_qubits, n_layers=L_max, seed=42)
        test_seqs = generate_real_amplitudes_dataset(
            n_sequences=args.n_test, n_qubits=args.n_qubits, n_layers=L_max, seed=123)

    # Apply the same "hard regime" style training-time truncation used for
    # Trotter (Table IV, generalized to arbitrary L_max), so the model also
    # learns to extrapolate from partial-depth training sequences here.
    train_seqs = []
    for seq in train_full:
        L_use = sample_max_length_train_generic(rng, args.partial_training_rate, L_max)
        train_seqs.append(truncate_sequence(seq, L_use))

    return train_seqs, test_seqs, L_max, "Ansatz layer"


TASKS = {
    "trotter": build_trotter_dataset,
    "real_amplitudes": build_real_amplitudes_dataset,
}


# ----------------------------------------------------------------------
# Shared, task-agnostic training + evaluation loop
# ----------------------------------------------------------------------
def to_tensors(seq, L_max):
    specs = torch.tensor(seq.spec_features(L_max), dtype=torch.float32).unsqueeze(0)  # (1,L,spec_dim)
    noisy_y = torch.tensor(seq.y_noisy, dtype=torch.float32).unsqueeze(0)             # (1,L)
    p_hat = torch.tensor(seq.p_hat, dtype=torch.float32).unsqueeze(0)                 # (1,L)
    y_true = torch.tensor(seq.y_noiseless, dtype=torch.float32).unsqueeze(0)          # (1,L)
    return specs, noisy_y, p_hat, y_true


def train(model, train_seqs, L_max, epochs=EPOCHS, lr=LR):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print("\nTraining NNAS ...")
    for epoch in range(epochs):
        perm = np.random.permutation(len(train_seqs))
        total_loss = 0.0
        for idx in perm:
            seq = train_seqs[idx]
            specs, noisy_y, p_hat, y_true = to_tensors(seq, L_max)

            opt.zero_grad()
            y_em, _ = model(specs, noisy_y, p_hat)
            loss = loss_fn(y_em, y_true)
            loss.backward()
            # Gradient clipping as a second safety net alongside the
            # softplus-constrained denominator in NNASCore -- see the
            # numerical-stability fix history for why this matters.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  avg train MSE = "
                  f"{total_loss / len(train_seqs):.6f}")


def evaluate(model, test_seqs, L_max, layer_label):
    model.eval()
    n_test = len(test_seqs)
    abs_err_noisy = np.full((n_test, L_max), np.nan)
    abs_err_nnas = np.full((n_test, L_max), np.nan)

    with torch.no_grad():
        for i, seq in enumerate(test_seqs):
            specs, noisy_y, p_hat, y_true = to_tensors(seq, L_max)
            y_em, _ = model(specs, noisy_y, p_hat)

            abs_err_noisy[i, :seq.L] = np.abs(seq.y_noisy - seq.y_noiseless)
            abs_err_nnas[i, :seq.L] = np.abs(y_em.squeeze(0).numpy() - seq.y_noiseless)

    mae_noisy_per_layer = np.nanmean(abs_err_noisy, axis=0)
    mae_nnas_per_layer = np.nanmean(abs_err_nnas, axis=0)

    print("\n" + "=" * 70)
    print(f"{layer_label:>12} | {'MAE (Noisy)':>12} | {'MAE (NNAS)':>12} | "
          f"{'Rel. reduction':>15}")
    print("-" * 70)
    for l in range(L_max):
        mn, mnn = mae_noisy_per_layer[l], mae_nnas_per_layer[l]
        if np.isnan(mn):
            continue
        red = 100 * (1 - mnn / mn)
        print(f"{l + 1:>12d} | {mn:>12.5f} | {mnn:>12.5f} | {red:>14.1f}%")

    overall_noisy = np.nanmean(abs_err_noisy)
    overall_nnas = np.nanmean(abs_err_nnas)
    deep_start = int(0.75 * L_max)
    deep_noisy = np.nanmean(abs_err_noisy[:, deep_start:])
    deep_nnas = np.nanmean(abs_err_nnas[:, deep_start:])

    print("=" * 70)
    print(f"Overall MAE                 : Noisy = {overall_noisy:.5f} | NNAS = {overall_nnas:.5f} "
          f"| reduction = {100 * (1 - overall_nnas / overall_noisy):.1f}%")
    print(f"Deepest quartile of layers  : Noisy = {deep_noisy:.5f} | NNAS = {deep_nnas:.5f} "
          f"| reduction = {100 * (1 - deep_nnas / deep_noisy):.1f}%")
    print("=" * 70)

    assert overall_nnas < overall_noisy, "Sanity check failed: NNAS did not reduce MAE!"
    print("\n[OK] NNAS mitigation reduces MAE relative to the noisy baseline.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS.keys()), default="trotter")
    parser.add_argument("--data", type=str, default=None,
                         help="[real_amplitudes only] path to a dataset saved by "
                              "real_amplitudes_dataset.generate_layerwise_data(...). "
                              "If omitted, a synthetic dataset is generated instead.")
    parser.add_argument("--n_qubits", type=int, default=4,
                         help="[real_amplitudes only]")
    parser.add_argument("--n_layers", type=int, default=10,
                         help="[real_amplitudes only]")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--partial_training_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_seqs, test_seqs, L_max, layer_label = TASKS[args.task](args)

    spec_dim = train_seqs[0].spec_features(L_max).shape[-1]
    model = NNASForQEM(spec_dim=spec_dim, hidden_dim=32, d=8, use_noisy_results=True)

    train(model, train_seqs, L_max)
    evaluate(model, test_seqs, L_max, layer_label)


if __name__ == "__main__":
    main()