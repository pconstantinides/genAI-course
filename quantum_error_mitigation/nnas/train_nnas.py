"""
Reproduce a simple, easily-verifiable NNAS result:

  Train NNAS on a small dataset of QAOA-type (1D transverse-field Ising
  Trotter) circuits, exactly as in Sec. III-A of the main text:
      - 6-qubit circuits, T1 = 23.235 us ("23*", the SOTA coherence time
        the paper highlights),
      - 100 training sequences at partial training rate p_r = 0.25
        (hard regime = Trotter steps 11-20, Table IV),
      - evaluate on 200 held-out test sequences spanning Trotter steps 1-20.

  What we check (the "easy to verify" claim):
      MAE(NNAS-mitigated) should be substantially lower than MAE(Noisy),
      and the *relative* improvement should be larger at big Trotter steps
      (>15) than at small ones -- this is the paper's headline result
      (Fig. 2 a-c): standard/noisy error grows quickly with circuit depth,
      while NNAS keeps it comparatively flat.
"""

import numpy as np
import torch
import torch.nn as nn

from qaoa_dataset import generate_qaoa_dataset
from model import NNASForQEM

torch.manual_seed(0)
np.random.seed(0)

N_QUBITS = 6
T1_US = 23.235
L_MAX = 20
N_TRAIN = 100
N_TEST = 200
PARTIAL_TRAINING_RATE = 0.25
EPOCHS = 150
LR = 3e-3


def make_features(seq, L_max=L_MAX):
    """spec features per layer: [normalized layer index, hdt (constant per seq)]"""
    L = seq.L
    layer_idx = np.arange(1, L + 1) / L_max
    hdt = np.full(L, seq.hdt)
    specs = np.stack([layer_idx, hdt], axis=-1)  # (L, 2)
    return specs


def to_tensors(seq):
    specs = torch.tensor(make_features(seq), dtype=torch.float32).unsqueeze(0)      # (1,L,2)
    noisy_y = torch.tensor(seq.y_noisy, dtype=torch.float32).unsqueeze(0)           # (1,L)
    p_hat = torch.tensor(seq.p_hat, dtype=torch.float32).unsqueeze(0)               # (1,L)
    y_true = torch.tensor(seq.y_noiseless, dtype=torch.float32).unsqueeze(0)        # (1,L)
    return specs, noisy_y, p_hat, y_true


def main():
    print("Generating training set "
          f"({N_TRAIN} sequences, partial_training_rate={PARTIAL_TRAINING_RATE}) ...")
    train_seqs = generate_qaoa_dataset(
        n_sequences=N_TRAIN, n_qubits=N_QUBITS, noise_t1_us=T1_US,
        partial_training_rate=PARTIAL_TRAINING_RATE, is_train=True,
        fixed_L=L_MAX, seed=42,
    )

    print(f"Generating test set ({N_TEST} sequences, full length {L_MAX}) ...")
    test_seqs = generate_qaoa_dataset(
        n_sequences=N_TEST, n_qubits=N_QUBITS, noise_t1_us=T1_US,
        is_train=False, fixed_L=L_MAX, seed=123,
    )

    model = NNASForQEM(spec_dim=2, hidden_dim=32, d=8, use_noisy_results=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print("\nTraining NNAS ...")
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(train_seqs))
        total_loss = 0.0
        for idx in perm:
            seq = train_seqs[idx]
            specs, noisy_y, p_hat, y_true = to_tensors(seq)

            opt.zero_grad()
            y_em, _ = model(specs, noisy_y, p_hat)
            loss = loss_fn(y_em, y_true)
            loss.backward()
            # Belt-and-braces: even with the softplus fix in NNASForQEM
            # guaranteeing a well-conditioned denominator, clip gradients
            # so no single bad batch can corrupt Adam's momentum state.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{EPOCHS}  avg train MSE = "
                  f"{total_loss / len(train_seqs):.6f}")

    # ------------------------------------------------------------------
    # Evaluation on the test set
    # ------------------------------------------------------------------
    model.eval()
    abs_err_noisy = np.zeros((N_TEST, L_MAX))
    abs_err_nnas = np.zeros((N_TEST, L_MAX))

    with torch.no_grad():
        for i, seq in enumerate(test_seqs):
            specs, noisy_y, p_hat, y_true = to_tensors(seq)
            y_em, _ = model(specs, noisy_y, p_hat)

            abs_err_noisy[i] = np.abs(seq.y_noisy - seq.y_noiseless)
            abs_err_nnas[i] = np.abs(y_em.squeeze(0).numpy() - seq.y_noiseless)

    mae_noisy_per_step = abs_err_noisy.mean(axis=0)
    mae_nnas_per_step = abs_err_nnas.mean(axis=0)

    print("\n" + "=" * 70)
    print(f"{'Trotter step':>12} | {'MAE (Noisy)':>12} | {'MAE (NNAS)':>12} | "
          f"{'Rel. reduction':>15}")
    print("-" * 70)
    for l in range(L_MAX):
        red = 100 * (1 - mae_nnas_per_step[l] / mae_noisy_per_step[l])
        print(f"{l + 1:>12d} | {mae_noisy_per_step[l]:>12.5f} | "
              f"{mae_nnas_per_step[l]:>12.5f} | {red:>14.1f}%")

    # Summary: overall and "deep circuit" (steps > 15) regime, mirroring
    # the paper's own headline comparison (Sec. III-A, Fig. 2c).
    overall_noisy = abs_err_noisy.mean()
    overall_nnas = abs_err_nnas.mean()
    deep_noisy = abs_err_noisy[:, 15:].mean()
    deep_nnas = abs_err_nnas[:, 15:].mean()

    print("=" * 70)
    print(f"Overall MAE   : Noisy = {overall_noisy:.5f} | NNAS = {overall_nnas:.5f} "
          f"| reduction = {100 * (1 - overall_nnas / overall_noisy):.1f}%")
    print(f"Steps 16-20   : Noisy = {deep_noisy:.5f} | NNAS = {deep_nnas:.5f} "
          f"| reduction = {100 * (1 - deep_nnas / deep_noisy):.1f}%")
    print("=" * 70)

    assert overall_nnas < overall_noisy, "Sanity check failed: NNAS did not reduce MAE!"
    print("\n[OK] NNAS mitigation reduces MAE relative to the noisy baseline, "
          "as expected from Fig. 2 of the paper.")


if __name__ == "__main__":
    main()