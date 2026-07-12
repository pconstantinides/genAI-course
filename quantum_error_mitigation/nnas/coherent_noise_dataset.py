"""
Datasets for the Dual-State NNAS research plan (Task 2): coherent
(systematic over-rotation / calibration drift) vs. stochastic
(depolarizing / amplitude-phase damping) error, and their combinations, on
the same 1D transverse-field Ising Trotter circuit used in qem_dataset.py.

Four conditions:
  A. stochastic : depolarizing + amplitude/phase-damping-derived Pauli
                  noise only (qem_dataset.NoiseLevel, plus an optional
                  extra standalone depolarizing component).
  B. coherent   : systematic Rx/Rz over-rotation (a constant, per-sequence
                  "calibration offset") only -- no stochastic channel.
  C. mixed      : A + B combined.
  D. drift      : layer-dependent coherent over-rotation,
                  eps_l = eps0 + alpha*(l-1) -- no stochastic channel.

All four produce `CoherentSequence` objects exposing the same (L,
y_noiseless, y_noisy, p_hat, spec_features) interface as qem_dataset.Sequence
and real_amplitudes_dataset.RASequence, so they plug directly into the same
training loop as the other circuit families.

IMPORTANT -- p_hat and coherent noise: Maximum Noise Decomposition (the
basis for p_hat, Supp. Sec. I of the original NNAS paper) only decomposes
CPTP *stochastic* channels; it has no notion of a coherent/unitary error.
So p_hat here is computed from the stochastic component ONLY, and is
identically zero for the pure-coherent datasets (B, D). Any correction for
coherent error must therefore come entirely from the learned r_hat term --
this is exactly what makes B/D a meaningful test of whether splitting the
recurrent state into stochastic/coherent branches helps: the mitigation
*output* formula is unchanged (Eq. 2), only the internal architecture that
predicts r_hat differs between Original NNAS and Dual-State NNAS.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qem_dataset import (
    DensitySim, NoiseLevel, layer_effective_rate,
    sample_max_length_train_generic,
)

CONDITIONS = ("stochastic", "coherent", "mixed", "drift")


# --------------------------------------------------------------------------
# Error configs
# --------------------------------------------------------------------------
@dataclass
class CoherentErrorConfig:
    eps_x0: float = 0.0   # baseline systematic over-rotation on every Rx gate (rad)
    eps_z0: float = 0.0   # baseline systematic over-rotation on every Rz gate (rad)
    alpha: float = 0.0    # linear drift rate: eps_l = eps0 + alpha*(l-1), l = 1-indexed layer

    def eps_x(self, layer_idx_1based: int) -> float:
        return self.eps_x0 + self.alpha * (layer_idx_1based - 1)

    def eps_z(self, layer_idx_1based: int) -> float:
        return self.eps_z0 + self.alpha * (layer_idx_1based - 1)


@dataclass
class StochasticErrorConfig:
    """Combines the existing T1/T2-derived Pauli channel (amplitude+phase
    damping, via qem_dataset.NoiseLevel) with an additional standalone
    depolarizing component -- satisfying Task 2.A's "depolarizing noise" +
    "amplitude damping" pair as two distinct stochastic contributions."""
    t1_us: float = 23.235
    extra_depolarizing_p: float = 0.0  # additional per-gate depolarizing prob


# --------------------------------------------------------------------------
# General-purpose Trotter simulator: stochastic and/or coherent error,
# independently switchable. A NEW function -- qem_dataset.py's own
# simulate_trotter_sequence is untouched, so the original Trotter pipeline
# is unaffected. Circuit structure (Rx layer, then CNOT-Rz-CNOT chain) is
# identical to qem_dataset.simulate_trotter_sequence.
# --------------------------------------------------------------------------
def simulate_trotter_sequence_general(
    n_qubits, J, h, dt, L,
    stochastic: StochasticErrorConfig = None,
    coherent: CoherentErrorConfig = None,
):
    theta_x = 2 * h * dt
    theta_z = -2 * J * dt

    sim_clean = DensitySim(n_qubits)
    sim_noisy = DensitySim(n_qubits)

    noise_level = NoiseLevel(stochastic.t1_us) if stochastic is not None else None
    if noise_level is not None:
        pX1, pY1, pZ1 = noise_level.single_qubit_rates()
        pX2, pY2, pZ2 = noise_level.two_qubit_component_rates()
    extra_dep = stochastic.extra_depolarizing_p if stochastic is not None else 0.0

    y_noiseless = np.zeros(L)
    y_noisy = np.zeros(L)
    last_q = n_qubits - 1

    for l in range(L):
        layer_idx = l + 1  # 1-based
        eps_x = coherent.eps_x(layer_idx) if coherent is not None else 0.0
        eps_z = coherent.eps_z(layer_idx) if coherent is not None else 0.0

        # --- column of Rx rotations ---
        for q in range(n_qubits):
            sim_clean.apply_rx(theta_x, q)
            sim_noisy.apply_rx(theta_x + eps_x, q)  # coherent over-rotation
            if noise_level is not None:
                sim_noisy.apply_pauli_channel_1q(pX1, pY1, pZ1, q)
            if extra_dep > 0:
                sim_noisy.apply_depolarizing_1q(extra_dep, q)

        # --- chain of Rzz gates: (0,1),(1,2),...,(n-2,n-1) ---
        for q in range(n_qubits - 1):
            c, t = q, q + 1
            sim_clean.apply_cnot(c, t)
            sim_noisy.apply_cnot(c, t)
            if noise_level is not None:
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, c)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, t)
            if extra_dep > 0:
                sim_noisy.apply_depolarizing_2q(extra_dep, c, t)

            sim_clean.apply_rz(theta_z, t)
            sim_noisy.apply_rz(theta_z + eps_z, t)  # coherent over-rotation
            if noise_level is not None:
                sim_noisy.apply_pauli_channel_1q(pX1, pY1, pZ1, t)
            if extra_dep > 0:
                sim_noisy.apply_depolarizing_1q(extra_dep, t)

            sim_clean.apply_cnot(c, t)
            sim_noisy.apply_cnot(c, t)
            if noise_level is not None:
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, c)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, t)
            if extra_dep > 0:
                sim_noisy.apply_depolarizing_2q(extra_dep, c, t)

        y_noiseless[l] = sim_clean.expval_Z(last_q)
        y_noisy[l] = sim_noisy.expval_Z(last_q)

    return y_noiseless, y_noisy


def compute_p_hat(n_qubits: int, L: int, stochastic: StochasticErrorConfig = None) -> np.ndarray:
    """MND-based per-layer prior, from the STOCHASTIC component only (see
    module docstring) -- identically zero when stochastic is None."""
    if stochastic is None:
        return np.zeros(L)

    noise_level = NoiseLevel(stochastic.t1_us)
    p_layer = layer_effective_rate(noise_level, n_qubits)

    if stochastic.extra_depolarizing_p > 0:
        n_single_gates = n_qubits + (n_qubits - 1)      # Rx's + Rz's
        n_two_qubit_gates = 2 * (n_qubits - 1)           # 2 CNOTs per RZZ block
        survival_extra = (1 - stochastic.extra_depolarizing_p) ** (n_single_gates + n_two_qubit_gates)
        p_layer = 1 - (1 - p_layer) * survival_extra

    return np.full(L, p_layer)


# --------------------------------------------------------------------------
# Sequence container (same interface as qem_dataset.Sequence)
# --------------------------------------------------------------------------
@dataclass
class CoherentSequence:
    hdt: float
    L: int
    y_noiseless: np.ndarray
    y_noisy: np.ndarray
    p_hat: np.ndarray
    eps_x0: float
    eps_z0: float
    alpha: float
    condition: str  # 'stochastic' | 'coherent' | 'mixed' | 'drift' (bookkeeping)

    def spec_features(self, L_max: int) -> np.ndarray:
        """[normalized layer index, hdt, eps_x0, eps_z0, alpha] -- eps_x0/
        eps_z0/alpha are treated as KNOWN, calibration-measurable
        quantities (analogous to the error rates fed as features for the
        other circuit families), not something the model must infer blind.
        Zero for datasets/conditions where that term is absent."""
        layer_idx = np.arange(1, self.L + 1) / L_max
        hdt_col = np.full(self.L, self.hdt)
        epsx_col = np.full(self.L, self.eps_x0)
        epsz_col = np.full(self.L, self.eps_z0)
        alpha_col = np.full(self.L, self.alpha)
        return np.stack([layer_idx, hdt_col, epsx_col, epsz_col, alpha_col], axis=-1)  # (L, 5)


def load_coherent_dataset(filename: str):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        return loaded.tolist()
    return list(loaded)


# --------------------------------------------------------------------------
# Dataset generator
# --------------------------------------------------------------------------
def generate_coherent_dataset(
    condition: str,
    n_sequences: int,
    n_qubits: int = 6,
    fixed_L: int = 20,
    is_train: bool = True,
    filename: str = None,
    load_if_exists: bool = True,
    partial_training_rate: float = 0.25,
    seed: int = 0,
    t1_us: float = 23.235,
    extra_depolarizing_p: float = 0.01,
    eps0_range=(0.03, 0.10),
    alpha_range=(0.006, 0.02),
    J_over_h: float = 0.6,
):
    """
    condition: one of CONDITIONS = ('stochastic', 'coherent', 'mixed', 'drift').
    is_train=True applies the same hard-regime length truncation as the
    Trotter dataset (Table IV, generalized via sample_max_length_train_generic).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")

    if filename is not None and load_if_exists:
        path = Path(filename)
        if path.exists():
            print(f"Loading existing dataset from {path}")
            return load_coherent_dataset(str(path))

    rng = np.random.default_rng(seed)

    use_stochastic = condition in ("stochastic", "mixed")
    use_coherent_const = condition in ("coherent", "mixed")
    use_drift = condition == "drift"

    sequences = []
    for _ in tqdm(range(n_sequences), desc=f"Generating {condition} dataset"):
        hdt = rng.uniform(0.5, 2.0)
        h_param = hdt
        J_param = J_over_h * h_param
        dt = 1.0

        eps_x0 = eps_z0 = alpha = 0.0
        if use_coherent_const:
            eps_x0 = rng.uniform(*eps0_range) * rng.choice([-1.0, 1.0])
            eps_z0 = rng.uniform(*eps0_range) * rng.choice([-1.0, 1.0])
        if use_drift:
            # small baseline offset + a genuine per-layer drift term
            eps_x0 = 0.3 * rng.uniform(*eps0_range) * rng.choice([-1.0, 1.0])
            eps_z0 = 0.3 * rng.uniform(*eps0_range) * rng.choice([-1.0, 1.0])
            alpha = rng.uniform(*alpha_range) * rng.choice([-1.0, 1.0])

        stochastic_cfg = (
            StochasticErrorConfig(t1_us=t1_us, extra_depolarizing_p=extra_depolarizing_p)
            if use_stochastic else None
        )
        coherent_cfg = (
            CoherentErrorConfig(eps_x0=eps_x0, eps_z0=eps_z0, alpha=alpha)
            if (use_coherent_const or use_drift) else None
        )

        L_use = fixed_L if not is_train else sample_max_length_train_generic(
            rng, partial_training_rate, fixed_L)

        y_clean_full, y_noisy_full = simulate_trotter_sequence_general(
            n_qubits, J_param, h_param, dt, fixed_L,
            stochastic=stochastic_cfg, coherent=coherent_cfg,
        )
        p_hat_full = compute_p_hat(n_qubits, fixed_L, stochastic_cfg)

        sequences.append(CoherentSequence(
            hdt=hdt, L=L_use,
            y_noiseless=y_clean_full[:L_use].copy(),
            y_noisy=y_noisy_full[:L_use].copy(),
            p_hat=p_hat_full[:L_use].copy(),
            eps_x0=eps_x0, eps_z0=eps_z0, alpha=alpha,
            condition=condition,
        ))
    
    if filename is not None:
        np.save(filename, sequences, allow_pickle=True)

    return sequences


if __name__ == "__main__":
    # Quick sanity check: per-layer MAE for each of the four conditions.
    # Expect: 'stochastic' MAE grows smoothly (decay-like, as in the
    # original Trotter dataset); 'coherent' MAE is present but doesn't
    # necessarily grow monotonically (systematic bias, not decay);
    # 'mixed' combines both; 'drift' MAE should grow with layer depth as
    # the accumulated over-rotation angle increases.
    
    from run_experiment import _dataset_path
    seed = 23
    n_sequences = 20_000
    
    for condition in CONDITIONS:
        seqs = generate_coherent_dataset(
            condition, n_sequences=n_sequences, n_qubits=4, fixed_L=20,
            is_train=True, seed=seed, filename=_dataset_path(condition, "train", n_sequences, seed)
        )
        # errs = np.array([np.abs(s.y_noisy - s.y_noiseless) for s in seqs])
        # p_hat_example = seqs[0].p_hat
        # print(f"{condition:>10} | per-layer MAE: {np.round(errs.mean(axis=0), 4)} "
        #       f"| p_hat[0]={p_hat_example[0]:.4f}")