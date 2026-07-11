"""
Dataset generation for Neural Noise Accumulation Surrogate (NNAS) experiments,
following Section III (main text) and Supplementary Section III / Fig. S2a:

  - 1D transverse-field Ising Trotterized ("QAOA-type") circuit:
        H_Ising = -J * sum_j Z_j Z_{j+1} + h * sum_j X_j
    first-order Trotter layer = [ Rx(2 h dt) on every qubit ]
                                then chain of Rzz(-2 J dt) gates on
                                neighbouring qubits (i, i+1), each Rzz
                                decomposed as CNOT - Rz(-2 J dt) - CNOT.

  - Layer-dependent Pauli noise derived from amplitude/phase damping
    (T1, T2), mapped via randomized compiling onto a Pauli channel
    (Supp. Sec. III.A, Eqs. 16-18 & Table II):
        pX = pY = (1 - exp(-t/T1)) / 4
        pZ = (1 - exp(-t/T2)) / 2 - (1 - exp(-t/T1)) / 4
    applied after every single-qubit gate (t = 18ns) and, independently,
    on both qubits touched by every CNOT (t = 48ns) -- this reproduces
    the two-qubit joint probabilities in Supp. Eq. list (since pX=pY,
    the "uncorrelated two independent single-qubit Pauli channels"
    picture is exactly equivalent to the paper's two-qubit formulas).

  - Effective per-layer rate p_hat_l is obtained via the backward-shift
    formula (Supp. Sec. I, Eq. 2):  p = 1 - prod_d (1 - p^d)
    over every gate d in the layer, using the single-qubit-gate-time
    Pauli rate for single-qubit gates and the two-qubit-gate-time
    combined rate (Table II, column b) for every CNOT.

The dataset produced mirrors Table VI / Sec. III-A-C of the Supplement:
sequences {y_tilde_l}_{l=1}^L (noisy), {y_l}_{l=1}^L (noiseless),
and {p_hat_j}_{j=1}^L (prior effectiveness rates) for QAOA-type circuits.
"""

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------
# Basic single/two-qubit operators
# --------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rx(theta):
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
                     dtype=complex)


# --------------------------------------------------------------------------
# Density-matrix simulator represented as a rank-2n tensor of shape
# (2,)*n (ket indices) + (2,)*n (bra indices). This lets us apply
# single/two-qubit unitaries and Pauli channels without ever building the
# full 2^n x 2^n matrix explicitly for the operator itself.
# --------------------------------------------------------------------------
class DensitySim:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        dim = 2 ** n_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0  # |00...0><00...0|
        self.rho = rho.reshape([2] * n_qubits + [2] * n_qubits)

    def _apply_unitary_1q(self, U, q):
        n = self.n
        rho = np.tensordot(U, self.rho, axes=([1], [q]))
        rho = np.moveaxis(rho, 0, q)
        rho = np.tensordot(rho, U.conj().T, axes=([n + q], [0]))
        rho = np.moveaxis(rho, -1, n + q)
        self.rho = rho

    def _apply_unitary_2q(self, U4, q0, q1):
        # U4: (2,2,2,2) tensor mapping (out0,out1,in0,in1)
        n = self.n
        rho = np.tensordot(U4, self.rho, axes=([2, 3], [q0, q1]))
        rho = np.moveaxis(rho, [0, 1], [q0, q1])
        U4d = U4.conj().transpose(2, 3, 0, 1)  # dagger, index order (in0,in1,out0,out1)
        rho = np.tensordot(rho, U4d, axes=([n + q0, n + q1], [0, 1]))
        rho = np.moveaxis(rho, [-2, -1], [n + q0, n + q1])
        self.rho = rho

    def apply_rx(self, theta, q):
        self._apply_unitary_1q(rx(theta), q)

    def apply_rz(self, theta, q):
        self._apply_unitary_1q(rz(theta), q)

    def apply_cnot(self, control, target):
        U4 = np.zeros((2, 2, 2, 2), dtype=complex)
        for c in range(2):
            for t in range(2):
                out_c, out_t = c, (t ^ c)
                U4[out_c, out_t, c, t] = 1.0
        self._apply_unitary_2q(U4, control, target)

    def apply_pauli_channel_1q(self, pX_, pY_, pZ_, q):
        """Single-qubit Pauli channel: (1-pX-pY-pZ) rho + pX X rho X + ..."""
        p0 = 1.0 - pX_ - pY_ - pZ_
        rho0 = self.rho
        acc = p0 * rho0
        for p, P in [(pX_, X), (pY_, Y), (pZ_, Z)]:
            if p == 0.0:
                continue
            self.rho = rho0
            self._apply_unitary_1q(P, q)
            acc = acc + p * self.rho
        self.rho = acc

    def expval_Z(self, q):
        n = self.n
        dim = 2 ** n
        rho_mat = self.rho.reshape(dim, dim)
        # Tr[rho Z_q] via diagonal trick: sign is +1 if bit q of index is 0, else -1
        diag = np.real(np.diag(rho_mat))
        idx = np.arange(dim)
        bit = (idx >> (n - 1 - q)) & 1
        sign = 1.0 - 2.0 * bit
        return float(np.sum(diag * sign))

    # ----------------------------------------------------------------
    # Additions for the RealAmplitudes / depolarizing-noise extension
    # (see real_amplitudes_dataset.py). Nothing above this point is
    # modified -- these are new, additive capabilities only.
    # ----------------------------------------------------------------
    def expval_Z_parity(self, qubits):
        """Tr[rho * prod_{q in qubits} Z_q] via the same diagonal trick as
        expval_Z, generalized to a subset of qubits (global parity when
        `qubits` is every qubit)."""
        n = self.n
        dim = 2 ** n
        rho_mat = self.rho.reshape(dim, dim)
        diag = np.real(np.diag(rho_mat))
        idx = np.arange(dim)
        sign = np.ones(dim)
        for q in qubits:
            bit = (idx >> (n - 1 - q)) & 1
            sign = sign * (1.0 - 2.0 * bit)
        return float(np.sum(diag * sign))

    def apply_depolarizing_1q(self, p, q):
        """Standard single-qubit depolarizing channel:
             E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
        This is exactly `apply_pauli_channel_1q` with pX=pY=pZ=p/3, kept
        as a separate, self-documenting entry point for depolarizing-noise
        datasets."""
        self.apply_pauli_channel_1q(p / 3.0, p / 3.0, p / 3.0, q)

    def apply_depolarizing_2q(self, p, q0, q1):
        """Standard two-qubit depolarizing channel:
             E(rho) = (1-p) rho + (p/15) * sum_{(P,Q) != (I,I)} (P dot Q) rho (P dot Q)
        summed over all 16 combinations of single-qubit Paulis {I,X,Y,Z}
        on (q0, q1) except the identity pair. Each (P,Q) term is applied
        as two independent single-qubit unitary applications on a fresh
        copy of rho (P and Q commute since they act on different qubits),
        reusing `_apply_unitary_1q` -- no new tensor machinery needed."""
        if p == 0.0:
            return
        paulis = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
        names = ['I', 'X', 'Y', 'Z']
        rho0 = self.rho
        acc = (1.0 - p) * rho0
        weight = p / 15.0
        for name0 in names:
            for name1 in names:
                if name0 == 'I' and name1 == 'I':
                    continue
                self.rho = rho0
                self._apply_unitary_1q(paulis[name0], q0)
                self._apply_unitary_1q(paulis[name1], q1)
                acc = acc + weight * self.rho
        self.rho = acc


# --------------------------------------------------------------------------
# Noise-rate calculations (Supp. Sec III.A)
# --------------------------------------------------------------------------
SINGLE_Q_GATE_TIME = 18e-9   # seconds
TWO_Q_GATE_TIME = 48e-9      # seconds


def pauli_rates(t1, t2, gate_time):
    """Return (pX, pY, pZ) for a given gate execution time, from T1/T2."""
    pX_ = pY_ = (1 - np.exp(-gate_time / t1)) / 4
    pZ_ = (1 - np.exp(-gate_time / t2)) / 2 - (1 - np.exp(-gate_time / t1)) / 4
    return pX_, pY_, pZ_


def t2_from_t1(t1, t1_ref=23.2357e-6, t2_ref=15.6e-6):
    """Scale T2 proportionally with T1, keeping the reference ratio (Supp. III.A)."""
    return t1 * (t2_ref / t1_ref)


@dataclass
class NoiseLevel:
    t1_us: float

    @property
    def t1(self):
        return self.t1_us * 1e-6

    @property
    def t2(self):
        return t2_from_t1(self.t1)

    def single_qubit_rates(self):
        return pauli_rates(self.t1, self.t2, SINGLE_Q_GATE_TIME)

    def two_qubit_component_rates(self):
        """Per-qubit Pauli rates applied to each of the two qubits touched
        by a CNOT (uses the two-qubit gate time, Table II)."""
        return pauli_rates(self.t1, self.t2, TWO_Q_GATE_TIME)


# --------------------------------------------------------------------------
# Effective-rate (prior) estimation, Supp. Sec. I
# --------------------------------------------------------------------------
def layer_effective_rate(noise: NoiseLevel, n_qubits: int):
    """
    p_layer = 1 - prod_d (1 - p^d) over every gate d in one Trotter layer:
      - n_qubits single-qubit Rx gates (t = 18ns)
      - (n_qubits - 1) RZZ blocks, each = 2 CNOTs (t=48ns, combined via
        Table II's "two-qubit Pauli error rate", 1-(1-pX-pY-pZ)^2)
        + 1 single-qubit Rz gate (t = 18ns)
    """
    pX1, pY1, pZ1 = noise.single_qubit_rates()
    p_single = pX1 + pY1 + pZ1  # per single-qubit gate error rate

    pX2, pY2, pZ2 = noise.two_qubit_component_rates()
    p_single_at_2q_time = pX2 + pY2 + pZ2
    p_two = 1 - (1 - p_single_at_2q_time) ** 2  # Table II, column b

    n_single_gates = n_qubits + (n_qubits - 1)      # Rx's + Rz's
    n_two_qubit_gates = 2 * (n_qubits - 1)           # 2 CNOTs per RZZ block

    survival = (1 - p_single) ** n_single_gates * (1 - p_two) ** n_two_qubit_gates
    return 1.0 - survival


# --------------------------------------------------------------------------
# Full circuit simulation for one (J, h, dt) instance up to L Trotter steps
# --------------------------------------------------------------------------
def simulate_trotter_sequence(n_qubits, J, h, dt, L, noise: NoiseLevel = None):
    """
    Returns:
        y_noiseless: (L,) array of <Z_{n-1}> at each Trotter step (no noise)
        y_noisy:     (L,) array of <Z_{n-1}> at each Trotter step (noise applied),
                     or None if noise is None
    """
    theta_x = 2 * h * dt
    theta_z = -2 * J * dt

    sim_clean = DensitySim(n_qubits)
    sim_noisy = DensitySim(n_qubits) if noise is not None else None

    if noise is not None:
        pX1, pY1, pZ1 = noise.single_qubit_rates()
        pX2, pY2, pZ2 = noise.two_qubit_component_rates()

    y_noiseless = np.zeros(L)
    y_noisy = np.zeros(L) if noise is not None else None

    last_q = n_qubits - 1

    for l in range(L):
        # --- column of Rx rotations ---
        for q in range(n_qubits):
            sim_clean.apply_rx(theta_x, q)
            if noise is not None:
                sim_noisy.apply_rx(theta_x, q)
                sim_noisy.apply_pauli_channel_1q(pX1, pY1, pZ1, q)

        # --- chain of Rzz gates: (0,1),(1,2),...,(n-2,n-1) ---
        for q in range(n_qubits - 1):
            c, t = q, q + 1
            # CNOT
            sim_clean.apply_cnot(c, t)
            if noise is not None:
                sim_noisy.apply_cnot(c, t)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, c)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, t)
            # Rz on target
            sim_clean.apply_rz(theta_z, t)
            if noise is not None:
                sim_noisy.apply_rz(theta_z, t)
                sim_noisy.apply_pauli_channel_1q(pX1, pY1, pZ1, t)
            # CNOT
            sim_clean.apply_cnot(c, t)
            if noise is not None:
                sim_noisy.apply_cnot(c, t)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, c)
                sim_noisy.apply_pauli_channel_1q(pX2, pY2, pZ2, t)

        y_noiseless[l] = sim_clean.expval_Z(last_q)
        if noise is not None:
            y_noisy[l] = sim_noisy.expval_Z(last_q)

    return y_noiseless, y_noisy


# --------------------------------------------------------------------------
# Dataset assembly (Supp. Sec III.A-B, Table IV)
# --------------------------------------------------------------------------
@dataclass
class Sequence:
    hdt: float
    J: float
    h: float
    L: int                     # (possibly truncated) sequence length used
    y_noiseless: np.ndarray    # (L,)
    y_noisy: np.ndarray        # (L,)
    p_hat: np.ndarray          # (L,) effective rate per layer

    def spec_features(self, L_max: int) -> np.ndarray:
        """Per-layer circuit-specification features: [normalized layer
        index, hdt]. Used by the (task-agnostic) unified trainer so it
        doesn't need to know this is a Trotter sequence specifically --
        see real_amplitudes_dataset.RASequence.spec_features for the
        analogous method on the other supported circuit family."""
        layer_idx = np.arange(1, self.L + 1) / L_max
        hdt_col = np.full(self.L, self.hdt)
        return np.stack([layer_idx, hdt_col], axis=-1)  # (L, 2)


def truncate_sequence(seq, L_use: int):
    """Generic, task-agnostic helper: return a shallow-copied sequence
    (works for both Sequence and real_amplitudes_dataset.RASequence, since
    both expose the same L / y_noiseless / y_noisy / p_hat attributes)
    truncated to its first L_use layers."""
    import copy
    new_seq = copy.copy(seq)
    new_seq.L = L_use
    new_seq.y_noiseless = seq.y_noiseless[:L_use]
    new_seq.y_noisy = seq.y_noisy[:L_use]
    new_seq.p_hat = seq.p_hat[:L_use]
    return new_seq


def _sample_max_length_train(rng, p_r):
    """Table IV: selection rate for max Trotter step L in the training set."""
    r = rng.random()
    if r < (1 - p_r):
        return 10
    elif r < (1 - p_r) + p_r * 0.5:
        return rng.integers(11, 14)   # 11-13
    elif r < (1 - p_r) + p_r * 0.8:
        return rng.integers(14, 18)   # 14-17
    else:
        return rng.integers(18, 21)   # 18-20


def sample_max_length_train_generic(rng, p_r, L_max):
    """Generalization of _sample_max_length_train (Table IV) to an
    arbitrary L_max, expressed as fractions of L_max rather than
    Trotter-specific absolute step numbers (50% / 55-65% / 70-85% /
    90-100%, matching 10/11-13/14-17/18-20 out of L_max=20). Used by the
    unified trainer so the same hard-regime training scheme can be applied
    to any circuit family/depth, not just the L_max=20 Trotter case."""
    half = max(1, L_max // 2)
    r = rng.random()
    if r < (1 - p_r):
        return half
    elif r < (1 - p_r) + p_r * 0.5:
        lo, hi = half + 1, max(half + 2, int(round(0.65 * L_max)) + 1)
        return rng.integers(lo, hi) if hi > lo else L_max
    elif r < (1 - p_r) + p_r * 0.8:
        lo, hi = max(half + 2, int(round(0.65 * L_max)) + 1), max(half + 3, int(round(0.85 * L_max)) + 1)
        return rng.integers(lo, hi) if hi > lo else L_max
    else:
        lo, hi = max(half + 3, int(round(0.85 * L_max)) + 1), L_max + 1
        return rng.integers(lo, hi) if hi > lo else L_max


def generate_qaoa_dataset(
    n_sequences: int,
    n_qubits: int = 6,
    noise_t1_us: float = 23.235,
    partial_training_rate: float = 0.25,
    is_train: bool = True,
    fixed_L: int = 20,
    J_over_h: float = 0.6,
    seed: int = 0,
):
    """
    Generates a list of `Sequence` objects.

    is_train=True  -> lengths truncated per Table IV's hard-regime scheme
                       (training set has variable-length sequences).
    is_train=False -> every sequence uses the full fixed_L (test set, Sec III-A
                       uses 200 sequences spanning Trotter steps 1-20).
    """
    rng = np.random.default_rng(seed)
    noise = NoiseLevel(noise_t1_us)
    p_hat_full = np.array([layer_effective_rate(noise, n_qubits)] * fixed_L)

    sequences = []
    for _ in range(n_sequences):
        hdt = rng.uniform(0.5, 2.0)
        h = hdt / 1.0  # arbitrary absolute scale; only h*dt, J*dt matter physically
        # fix dt = 1 for simplicity (only the product h*dt, J*dt enters the circuit)
        dt = 1.0
        h_param = hdt
        J_param = J_over_h * h_param

        L_use = fixed_L if not is_train else _sample_max_length_train(rng, partial_training_rate)

        y_clean_full, y_noisy_full = simulate_trotter_sequence(
            n_qubits, J_param, h_param, dt, fixed_L, noise=noise
        )

        seq = Sequence(
            hdt=hdt, J=J_param, h=h_param, L=L_use,
            y_noiseless=y_clean_full[:L_use].copy(),
            y_noisy=y_noisy_full[:L_use].copy(),
            p_hat=p_hat_full[:L_use].copy(),
        )
        sequences.append(seq)

    return sequences


if __name__ == "__main__":
    # Quick sanity check: run a handful of sequences and print noisy vs
    # noiseless MAE growth with Trotter step (should increase with L,
    # and should increase as T1 decreases i.e. noise gets worse).
    for t1 in [20.0, 23.235, 40.0]:
        seqs = generate_qaoa_dataset(
            n_sequences=5, n_qubits=4, noise_t1_us=t1,
            is_train=False, fixed_L=10, seed=1,
        )
        errs = np.array([np.abs(s.y_noisy - s.y_noiseless) for s in seqs])
        print(f"T1={t1:>6.2f}us | per-step MAE:", np.round(errs.mean(axis=0), 4))