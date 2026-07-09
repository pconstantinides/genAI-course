"""
pauli_dataset.py

Generates the "Pauli" circuit dataset described in Placidi et al.,
"Deep Learning Approaches to Quantum Error Mitigation" (arXiv:2601.14226),
Section 2 and Algorithm 1.

For each circuit we produce:
    C       : circuit encoding tensor, shape (n_layers, n_qubits, 5),
              following the gate encoding of Table 1.
    B       : backend calibration feature vector, shape (101,),
              following Section 2.2.
    Pnoisy  : noisy output probability distribution, shape (32,)
    Pideal  : ideal (noiseless) output probability distribution, shape (32,)

Notes on fidelity to the paper:
  - Circuits are built exactly per Algorithm 1 (random Pauli gadgets
    exp(-i*alpha*P), P in {I,X,Y,Z}^n).
  - Circuits are compiled onto the IBM native gateset {X, SX, Rz, CX}
    (Section 2.1), as tket would do; we use Qiskit's transpiler here
    since tket is not assumed to be installed.
  - Real access to `ibm_algiers` calibration data via the IBM API is not
    assumed. `sample_backend_calibration` synthesizes plausible calibration
    snapshots (T1, T2, gate/readout errors) with a similar statistical
    profile to Fig. 1 of the paper. Replace this function with real
    `backend.properties()` calls if IBM hardware access is available.
  - Pnoisy is obtained from a noisy Aer simulation (density-matrix-equivalent
    via shot sampling with a thermal-relaxation + depolarizing + readout
    noise model built from B), matching the "Simulated" dataset described
    in Section 2.3.
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from multiprocessing import get_context

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    thermal_relaxation_error,
    depolarizing_error,
    ReadoutError,
)
from qiskit_ibm_runtime.fake_provider import FakeAlgiers

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_QUBITS = 5
N_OUTCOMES = 2 ** N_QUBITS          # 32, matches Pnoisy/Pideal dimension in paper
NATIVE_GATES = ["x", "sx", "rz", "cx"]
GATE_ENCODING_DIM = 5                # Table 1 encoding vector length
N_BACKEND_PARAMS = 101                # Section 2.2
PAULI_LETTERS = ["I", "X", "Y", "Z"]


# ---------------------------------------------------------------------------
# Algorithm 1: Pauli circuit generation protocol
# ---------------------------------------------------------------------------

def random_pauli_string(n_qubits: int, rng: np.random.Generator) -> str:
    """Uniform random Pauli string of length n_qubits (Sigma = {I,X,Y,Z}^N)."""
    return "".join(rng.choice(PAULI_LETTERS) for _ in range(n_qubits))


def append_pauli_gadget(circuit: QuantumCircuit, pauli: str, angle: float) -> None:
    """
    Appends exp(-i * angle * P) to `circuit`, where P is the tensor-product
    Pauli string `pauli` (e.g. "IXYZI"), using the standard
    basis-change + CNOT-ladder + Rz + uncompute construction.
    """
    active_qubits = [i for i, p in enumerate(pauli) if p != "I"]
    if not active_qubits:
        return  # exp(-i*angle*I) is a global phase; no circuit action needed

    # Basis change into the Z basis.
    for i, p in enumerate(pauli):
        if p == "X":
            circuit.h(i)
        elif p == "Y":
            circuit.sdg(i)
            circuit.h(i)

    # CNOT ladder accumulates parity onto the last active qubit.
    for a, b in zip(active_qubits[:-1], active_qubits[1:]):
        circuit.cx(a, b)

    circuit.rz(2 * angle, active_qubits[-1])

    for a, b in reversed(list(zip(active_qubits[:-1], active_qubits[1:]))):
        circuit.cx(a, b)

    # Undo the basis change.
    for i, p in enumerate(pauli):
        if p == "X":
            circuit.h(i)
        elif p == "Y":
            circuit.h(i)
            circuit.s(i)


def generate_pauli_circuit(n_qubits: int, T: int, rng: np.random.Generator) -> QuantumCircuit:
    """
    Algorithm 1 from the paper: build a circuit of T random Pauli gadgets,
    then measure all qubits.
    """
    qc = QuantumCircuit(n_qubits, n_qubits)
    for _ in range(T):
        pauli = random_pauli_string(n_qubits, rng)
        angle = rng.uniform(0, 2 * np.pi)
        append_pauli_gadget(qc, pauli, angle)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ---------------------------------------------------------------------------
# Compilation onto the IBM native gateset (Section 2.1)
# ---------------------------------------------------------------------------

def compile_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Compile onto {X, SX, Rz, CX} with linear qubit connectivity."""
    coupling = [[i, i + 1] for i in range(N_QUBITS - 1)] + \
               [[i + 1, i] for i in range(N_QUBITS - 1)]
    return transpile(
        qc,
        basis_gates=NATIVE_GATES,
        coupling_map=coupling,
        optimization_level=0,   # paper: "no further optimization is performed"
    )


# ---------------------------------------------------------------------------
# Circuit encoding (Table 1)
# ---------------------------------------------------------------------------

def _instruction_layers(qc_compiled: QuantumCircuit) -> List[List[Tuple[str, list, list]]]:
    """
    Greedily groups gate instructions of a compiled circuit into layers of
    parallel operations (a new layer starts whenever a qubit already used in
    the current layer is touched again).
    """
    layers: List[List[Tuple[str, list, list]]] = []
    current_layer: List[Tuple[str, list, list]] = []
    used_qubits = set()

    def qubit_idx(q):
        return qc_compiled.find_bit(q).index

    for instr in qc_compiled.data:
        name = instr.operation.name
        if name in ("measure", "barrier"):
            continue
        qubits = [qubit_idx(q) for q in instr.qubits]
        if any(q in used_qubits for q in qubits):
            layers.append(current_layer)
            current_layer = []
            used_qubits = set()
        current_layer.append((name, qubits, list(instr.operation.params)))
        used_qubits.update(qubits)

    if current_layer:
        layers.append(current_layer)
    return layers


def count_layers(qc_compiled: QuantumCircuit) -> int:
    return len(_instruction_layers(qc_compiled))


def encode_circuit(qc_compiled: QuantumCircuit, max_layers: int) -> np.ndarray:
    """
    Encodes a compiled circuit into an array Carray of shape
    (max_layers, N_QUBITS, GATE_ENCODING_DIM), following Table 1:

        X                -> (1, 0, 0, 0, 0)
        SX               -> (0, 1, 0, 0, 0)
        Rz(alpha)         -> (0, 0, (alpha mod 2)/2, 0, 0)   [alpha in units of pi]
        CX (control)      -> (0, 0, 0, -(1+target_idx), 0)
        CX (target)        -> (0, 0, 0,  1+control_idx, 0)
        idle / padding    -> (0, 0, 0, 0, 0)

    Circuits shorter than `max_layers` are zero-padded at the end.
    """
    layers = _instruction_layers(qc_compiled)
    arr = np.zeros((max_layers, N_QUBITS, GATE_ENCODING_DIM), dtype=np.float32)

    for l, layer in enumerate(layers):
        if l >= max_layers:
            break
        for name, qubits, params in layer:
            if name == "x":
                arr[l, qubits[0]] = [1, 0, 0, 0, 0]
            elif name == "sx":
                arr[l, qubits[0]] = [0, 1, 0, 0, 0]
            elif name == "rz":
                alpha = float(params[0]) / np.pi  # convert to units of pi
                arr[l, qubits[0]] = [0, 0, (alpha % 2) / 2.0, 0, 0]
            elif name == "cx":
                ctrl, targ = qubits
                arr[l, ctrl] = [0, 0, 0, -(1 + targ), 0]
                arr[l, targ] = [0, 0, 0, 1 + ctrl, 0]
    return arr


# ---------------------------------------------------------------------------
# Backend calibration data (Section 2.2, Fig. 1)
# ---------------------------------------------------------------------------

@dataclass
class BackendCalibration:
    t1: np.ndarray                       # (5,)  microseconds
    t2: np.ndarray                       # (5,)  microseconds
    freq: np.ndarray                     # (5,)  GHz
    readout_error: np.ndarray            # (5,)
    single_qubit_gate_error: np.ndarray  # (5,)
    single_qubit_gate_time: np.ndarray   # (5,)  ns
    cx_error: np.ndarray                 # (4,)  edges (0,1)(1,2)(2,3)(3,4)
    cx_time: np.ndarray                  # (4,)  ns

    def to_vector(self) -> np.ndarray:
        parts = [
            self.t1, self.t2, self.freq, self.readout_error,
            self.single_qubit_gate_error, self.single_qubit_gate_time,
            self.cx_error, self.cx_time,
        ]
        v = np.concatenate(parts).astype(np.float32)
        if len(v) < N_BACKEND_PARAMS:
            v = np.concatenate([v, np.zeros(N_BACKEND_PARAMS - len(v), dtype=np.float32)])
        return v[:N_BACKEND_PARAMS]


def sample_backend_calibration(rng: np.random.Generator) -> BackendCalibration:
    """
    Synthetic stand-in for an `ibm_algiers`-style calibration snapshot
    (Fig. 1b/1c). Replace with real `backend.properties()` data for
    genuine hardware calibration.
    """
    t1 = rng.normal(150, 40, size=5).clip(20, 350)
    t2 = rng.normal(120, 60, size=5).clip(10, 400)
    freq = rng.normal(5.0, 0.1, size=5).clip(4.5, 5.5)
    readout_error = rng.lognormal(mean=np.log(0.02), sigma=0.5, size=5).clip(0.001, 0.15)
    single_qubit_gate_error = rng.lognormal(mean=np.log(3e-4), sigma=0.5, size=5).clip(1e-5, 1e-2)
    single_qubit_gate_time = rng.normal(35, 5, size=5).clip(20, 60)
    cx_error = rng.lognormal(mean=np.log(8e-3), sigma=0.6, size=4).clip(1e-3, 5e-1)
    cx_time = rng.normal(300, 60, size=4).clip(150, 600)
    return BackendCalibration(
        t1, t2, freq, readout_error,
        single_qubit_gate_error, single_qubit_gate_time,
        cx_error, cx_time,
    )


def build_noise_model(cal: BackendCalibration | Backend) -> NoiseModel:
    """Builds a thermal-relaxation + depolarizing + readout NoiseModel from B."""
    if isinstance(cal, Backend):
        return NoiseModel.from_backend(cal)
    nm = NoiseModel()

    for q in range(N_QUBITS):
        t1_ns = cal.t1[q] * 1000.0
        t2_ns = min(cal.t2[q] * 1000.0, 2 * t1_ns * 0.999)
        gate_time = cal.single_qubit_gate_time[q]
        therm = thermal_relaxation_error(t1_ns, t2_ns, gate_time)
        depol = depolarizing_error(cal.single_qubit_gate_error[q], 1)
        err = therm.compose(depol)
        nm.add_quantum_error(err, ["x", "sx"], [q])
        nm.add_quantum_error(depol, ["rz"], [q])
        ro = cal.readout_error[q]
        nm.add_readout_error(
            ReadoutError([[1 - ro, ro], [ro, 1 - ro]]), [q]
        )

    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for k, (a, b) in enumerate(edges):
        gate_time = cal.cx_time[k]
        t1a_ns, t1b_ns = cal.t1[a] * 1000.0, cal.t1[b] * 1000.0
        t2a_ns = min(cal.t2[a] * 1000.0, 2 * t1a_ns * 0.999)
        t2b_ns = min(cal.t2[b] * 1000.0, 2 * t1b_ns * 0.999)
        therm2 = thermal_relaxation_error(t1a_ns, t2a_ns, gate_time).expand(
            thermal_relaxation_error(t1b_ns, t2b_ns, gate_time)
        )
        depol2 = depolarizing_error(cal.cx_error[k], 2)
        err2 = therm2.compose(depol2)
        nm.add_quantum_error(err2, "cx", [a, b])
        nm.add_quantum_error(err2, "cx", [b, a])

    return nm


# ---------------------------------------------------------------------------
# Simulating Pideal and Pnoisy (Section 2.3)
# ---------------------------------------------------------------------------

def ideal_probabilities(qc_compiled: QuantumCircuit) -> np.ndarray:
    qc_no_meas = qc_compiled.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(qc_no_meas)
    return sv.probabilities().astype(np.float32)


def noisy_probabilities(
    qc_compiled: QuantumCircuit, noise_model: NoiseModel, shots: int = 20000
) -> np.ndarray:
    # max_parallel_threads=1: when this function runs inside a
    # multiprocessing worker (see build_pauli_simulated_dataset), we want
    # process-level parallelism only. Letting each Aer instance also spawn
    # its own internal OpenMP threads would oversubscribe CPU cores when
    # n_workers processes are running concurrently.
    sim = AerSimulator(noise_model=noise_model, max_parallel_threads=1)
    tqc = transpile(qc_compiled, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    probs = np.zeros(N_OUTCOMES, dtype=np.float32)
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        probs[idx] = count / shots
    return probs


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

@dataclass
class PauliDatasetConfig:
    T_values: Tuple[int, ...] = (3, 4, 5, 6, 7, 9)  # Simulated Pauli T values, Sec. 2.3
    n_circuits_per_T: int = 200     # paper: 8,000 for Simulated; reduced here for tractability
    n_repeats: int = 3              # paper: 3 repeats per circuit
    shots: int = 20000              # paper: 20,000 shots/circuit
    seed: int = 0
    n_workers: int = 3              # multiprocessing.Pool size for the noisy-simulation stage


# ---------------------------------------------------------------------------
# Parallel worker: generates one (circuit, repeat) sample
#
# Circuit generation/compilation is cheap and inherently sequential (it
# consumes a shared RNG stream), so it stays on the main process. The
# expensive step -- sampling a backend calibration, building its noise
# model, and running the noisy Aer simulation -- is what gets distributed
# across the process pool. Each task is seeded independently (via
# np.random.SeedSequence.spawn) so results stay reproducible regardless of
# how many workers are used or how tasks get scheduled between them.
# ---------------------------------------------------------------------------

def _generate_one_repeat(payload) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qc_c, max_layers, shots, seed_seq = payload
    rng = np.random.default_rng(seed_seq)

    carray = encode_circuit(qc_c, max_layers=max_layers)
    pideal = ideal_probabilities(qc_c)

    cal = sample_backend_calibration(rng)
    nm = build_noise_model(FakeAlgiers()) # cal
    pnoisy = noisy_probabilities(qc_c, nm, shots=shots)

    return carray, cal.to_vector(), pnoisy, pideal


def build_pauli_simulated_dataset(config: PauliDatasetConfig = PauliDatasetConfig()) -> dict:
    """
    Builds a "Pauli / Simulated / ibm_algiers-style" dataset as described in
    Sections 2.1-2.3. Returns a dict with keys "C", "B", "Pnoisy", "Pideal".

    The per-(circuit, repeat) noisy-simulation work is distributed across a
    multiprocessing.Pool of `config.n_workers` processes.
    """
    rng = np.random.default_rng(config.seed)

    # Circuit generation & compilation stay single-process (fast, and the
    # random Pauli-gadget draws must come from one consistent RNG stream).
    raw_circuits: List[QuantumCircuit] = []
    for T in config.T_values:
        for _ in range(config.n_circuits_per_T):
            qc = generate_pauli_circuit(N_QUBITS, T, rng)
            qc_c = compile_circuit(qc)
            raw_circuits.append(qc_c)

    # Determine padding length = depth of the deepest circuit (largest T), as in
    # the paper ("Carray is padded ... to be the same shape as the Carray of
    # the maximal depth circuit with the largest T value").
    layer_counts = [count_layers(qc_c) for qc_c in raw_circuits]
    max_layers = max(layer_counts)

    # Build the full task list: one task per (circuit, repeat).
    n_tasks = len(raw_circuits) * config.n_repeats
    child_seeds = np.random.SeedSequence(config.seed).spawn(n_tasks)

    tasks = []
    seed_idx = 0
    for qc_c in raw_circuits:
        for _ in range(config.n_repeats):
            tasks.append((qc_c, max_layers, config.shots, child_seeds[seed_idx]))
            seed_idx += 1

    if config.n_workers and config.n_workers > 1:
        # NOTE: we explicitly use the "spawn" start method rather than the
        # platform default ("fork" on Linux). Forking a process after
        # Qiskit Aer has initialized its internal OpenMP thread pool can
        # deadlock the child processes; "spawn" starts each worker as a
        # fresh interpreter and avoids this entirely.
        ctx = get_context("spawn")
        with ctx.Pool(processes=config.n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_generate_one_repeat, tasks),
                    total=len(tasks),
                    desc="Generating dataset",
                )
            )
    else:
        results = [
            _generate_one_repeat(t)
            for t in tqdm(tasks, total=len(tasks), desc="Generating dataset")
        ]

    C_list, B_list, Pnoisy_list, Pideal_list = zip(*results)

    return {
        "C": np.stack(C_list),
        "B": np.stack(B_list),
        "Pnoisy": np.stack(Pnoisy_list),
        "Pideal": np.stack(Pideal_list),
        "max_layers": max_layers,
    }


if __name__ == "__main__":
    # Small, fast smoke-test configuration, generated with a 3-process pool.
    # Increase n_circuits_per_T / shots to approach the paper's scale
    # (8,000 circuits/T, 20,000 shots/circuit, n_repeats=3).
    cfg = PauliDatasetConfig(n_circuits_per_T=4_000, n_repeats=1, shots=4000, n_workers=3)

    t0 = time.time()
    dataset = build_pauli_simulated_dataset(cfg)
    t1 = time.time()

    n_units = len(cfg.T_values) * cfg.n_circuits_per_T * cfg.n_repeats
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in dataset.items()})
    print(f"Generated {n_units} circuit-repeat units with n_workers={cfg.n_workers} "
          f"in {t1 - t0:.2f}s ({(t1 - t0) / n_units:.4f} s/unit)")

    np.savez_compressed(
        "pauli_simulated_dataset.npz",
        C=dataset["C"], B=dataset["B"],
        Pnoisy=dataset["Pnoisy"], Pideal=dataset["Pideal"],
    )
    print("Saved pauli_simulated_dataset.npz")