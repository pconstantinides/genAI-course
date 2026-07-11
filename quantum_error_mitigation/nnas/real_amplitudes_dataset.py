"""
This module is the dataset-generation script originally used to produce
flat (single-final-value) datasets for the regression-style QEM models,
EXTENDED with a new per-layer generator so the same underlying circuits
and noise models can also feed NNAS's sequential training.

Everything above the "NNAS EXTENSION" marker below is the original,
unmodified script (`generate_data`, `_generate_noise_model`,
`_build_dataset_tensors`, `generate_noise_injected_dataset_from_existing`,
etc.) -- kept exactly as-is so the other regression models that depend on
it keep working unchanged, and so both the regression models and NNAS are
trained on data coming from the *same* circuits/noise models.

Everything below the marker is new:
  - `generate_layerwise_data(...)`: same ansatz (`n_local(nqubits, 'ry',
    'cx', reps=R, entanglement='linear')`) and the same
    `_generate_noise_model(...)` depolarizing noise, but instead of a
    single final expectation value per sample, it measures the ideal and
    noisy expectation value at *every* layer depth l=1..R (i.e. re-running
    the estimator on the sub-circuit built with reps=l, for l=1..R,
    batched into a single Estimator.run() call per ideal/noisy pair). This
    is what NNAS needs: a per-layer sequence, exactly like the Trotter/
    QAOA circuits in qem_dataset.py, rather than one final value.
  - Per the task instructions, ZNE folding is ignored here: only the
    scale=1 (unfolded) noisy circuit is simulated per layer, which is also
    why this new function does not need `GlobalFoldingPass` (its import is
    guarded below so this module still loads without the private
    `qiskit_helpers` package, since only the *original* `generate_data`
    path needs it).
  - `RASequence`: a dataclass exposing the same (L, y_noiseless, y_noisy,
    p_hat, spec_features) interface as qem_dataset.Sequence, so a single,
    unified trainer (train_nnas.py) can consume sequences from either
    circuit family without caring which one it is.
  - `load_layerwise_dataset(...)` / `generate_real_amplitudes_dataset(...)`:
    turn a saved layer-wise dataset (or an in-process generation call) into
    a list of `RASequence` objects ready for training.
"""

from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch

from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import PassManager
from qiskit.circuit.library import n_local
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime import EstimatorV2 as Estimator

try:
    from utils import GlobalFoldingPass
except ImportError:
    GlobalFoldingPass = None  # only needed by the original generate_data()'s
                               # scale>1 folding path; the new layer-wise
                               # generator below only ever uses scale=1.


@dataclass
class DatasetGenerationConfig:
    use_noise_injection: bool = False
    noise_injection_cluster_size: int = 10
    noise_injection_noise_type: str = "gaussian"
    noise_injection_noise_amount: float = 0.01
    noise_injection_noise_distribution: str = "centered"


def _generate_noise_model(tq_err_rate, sq_err_rate):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(sq_err_rate, 1), instructions=['u1', 'u2', 'u3']
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(tq_err_rate, 2), instructions=['cx', 'cz']
    )
    return nm


def _as_feature_vector(values):
    if values is None:
        return np.asarray([0.0], dtype=np.float64)

    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False).reshape(-1)

    if isinstance(values, (list, tuple)):
        flattened = []
        for item in values:
            if isinstance(item, (list, tuple, np.ndarray)):
                flattened.extend(_as_feature_vector(item).tolist())
            else:
                flattened.append(float(np.asarray(item, dtype=np.float64).reshape(())))
        return np.asarray(flattened, dtype=np.float64).reshape(-1)

    return np.asarray([float(np.asarray(values, dtype=np.float64).reshape(()))], dtype=np.float64)


def _coerce_target(value):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.float64(0.0)
    return np.float64(arr[0])


def _build_dataset_tensors(
    samples,
    nqubits: int,
    seed: int,
    error_rates=None,
    sample_error_rates=None,
    sample_single_qubit_error_rates=None,
    sample_noise_model_ids=None,
):
    if not samples:
        raise ValueError("No samples provided to build dataset tensors.")

    features = [_as_feature_vector(sample[1]) for sample in samples]
    feature_lengths = {feature.size for feature in features}
    if len(feature_lengths) > 1:
        raise ValueError("All samples must contain the same number of folded/noisy values.")

    inputs = np.vstack(features).astype(np.float64, copy=False)
    targets = np.asarray([_coerce_target(sample[0]) for sample in samples], dtype=np.float64)
    if error_rates is None:
        error_rates = np.logspace(-3, -1)

    error_rates = np.asarray(error_rates, dtype=np.float64).reshape(-1)
    metadata = {
        "num_samples": len(samples),
        "nqubits": nqubits,
        "seed": seed,
        "nfolds": int(inputs.shape[1]),
        "two_qubit_error_rates": error_rates.astype(float).tolist(),
    }

    if sample_error_rates is not None:
        metadata["sample_two_qubit_error_rates"] = np.asarray(sample_error_rates, dtype=np.float64).astype(float).tolist()
    if sample_single_qubit_error_rates is not None:
        metadata["sample_single_qubit_error_rates"] = np.asarray(sample_single_qubit_error_rates, dtype=np.float64).astype(float).tolist()
    if sample_noise_model_ids is not None:
        metadata["sample_noise_model_ids"] = np.asarray(sample_noise_model_ids, dtype=np.int64).astype(int).tolist()

    if torch is not None:
        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
            metadata,
        )

    return inputs, targets, metadata


def _sample_observable_values(base_value, config: DatasetGenerationConfig):
    cluster_size = int(config.noise_injection_cluster_size)
    if cluster_size <= 0:
        raise ValueError("noise_injection_cluster_size must be positive.")
    if config.noise_injection_noise_amount < 0:
        raise ValueError("noise_injection_noise_amount must be non-negative.")

    if config.noise_injection_noise_type == "gaussian":
        noise = np.random.normal(0.0, config.noise_injection_noise_amount, size=cluster_size)
    elif config.noise_injection_noise_type == "uniform":
        noise = np.random.uniform(
            -config.noise_injection_noise_amount,
            config.noise_injection_noise_amount,
            size=cluster_size,
        )
    elif config.noise_injection_noise_type == "laplace":
        noise = np.random.laplace(0.0, config.noise_injection_noise_amount, size=cluster_size)
    else:
        raise ValueError(
            "Unsupported noise_injection_noise_type. Choose from 'gaussian', 'uniform', or 'laplace'."
        )

    if config.noise_injection_noise_distribution == "biased":
        noise = noise + config.noise_injection_noise_amount
    elif config.noise_injection_noise_distribution != "centered":
        raise ValueError(
            "Unsupported noise_injection_noise_distribution. Choose from 'centered' or 'biased'."
        )

    if isinstance(base_value, np.ndarray):
        base_value = base_value.astype(np.float64, copy=False).reshape(-1)
        if base_value.size == 0:
            base_value = np.asarray([0.0], dtype=np.float64)
        base_values = np.repeat(base_value, cluster_size)
    else:
        base_values = np.full(cluster_size, float(np.asarray(base_value, dtype=np.float64).reshape(())))

    return base_values + noise


def generate_noise_injected_dataset_from_existing(
    source_dataset_path,
    output_file,
    generation_config: DatasetGenerationConfig = None,
    seed: int = 42,
):
    """Create a noise-injected dataset from an existing clean dataset file.

    The source dataset is expected to contain ``inputs`` and ``targets`` tensors,
    and each input sample is converted into a cluster of perturbed values. The
    original targets are preserved for each sample.
    """
    if generation_config is None:
        generation_config = DatasetGenerationConfig()

    np.random.seed(seed)
    torch.manual_seed(seed)

    source_path = Path(source_dataset_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    blob = torch.load(source_path, map_location="cpu")
    if not isinstance(blob, dict) or "inputs" not in blob or "targets" not in blob:
        raise ValueError("Source dataset must be a torch-save dict containing 'inputs' and 'targets'.")

    inputs = blob["inputs"]
    targets = blob["targets"]
    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs)
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    inputs = inputs.float().cpu().numpy()
    targets = targets.float().cpu().numpy().reshape(-1)
    if inputs.ndim == 1:
        inputs = inputs.reshape(-1, 1)

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("Source inputs and targets must describe the same number of samples.")

    injected_inputs = []
    for row in inputs:
        base_value = float(np.asarray(row, dtype=np.float64).reshape(-1)[0]) if row.size else 0.0
        injected_inputs.append(_sample_observable_values(base_value, generation_config))

    injected_inputs = np.vstack(injected_inputs).astype(np.float64, copy=False)
    metadata = dict(blob.get("metadata", {}))
    metadata["source_dataset_path"] = str(source_path)
    metadata["source_num_samples"] = int(inputs.shape[0])
    metadata["source_input_shape"] = list(inputs.shape)
    metadata["noise_injection_generation"] = True
    metadata["noise_injection_cluster_size"] = int(generation_config.noise_injection_cluster_size)
    metadata["noise_injection_noise_type"] = generation_config.noise_injection_noise_type
    metadata["noise_injection_noise_amount"] = float(generation_config.noise_injection_noise_amount)
    metadata["noise_injection_noise_distribution"] = generation_config.noise_injection_noise_distribution

    if output_file is not None:
        save_path = Path(output_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "inputs": torch.tensor(injected_inputs, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "metadata": metadata,
        }
        torch.save(payload, save_path)
        print(f"Saved noise-injected dataset to {save_path}")

    return injected_inputs, targets, metadata


def generate_data(
    num_samples: int,
    samples_per_nm: int,
    context: str,
    progress=True,
    seed=42,
    output_file="qem_dataset.pt",
    fold_scales=None,
    two_qubit_error_rates=None,
    generation_config: DatasetGenerationConfig = None,
):
    """Generates ideal and noisy measurements of a real-amplitudes ansatz and optionally saves them.

    Each generated sample is stored as ``(ideal_value, (fold_1, fold_2, ...))``.
    The folded values can be used as a feature vector for ZNE-style models.

    Args:
        num_samples (int): Total number of samples to generate.
        samples_per_nm (int): Number of samples per noise model.
        context (str): Either 'nocontext' or 'global_folding'.
        progress (bool): Whether to show a progress bar.
        seed (int): Random seed for reproducibility.
        output_file (str | None): Optional path to save the dataset in a Torch-friendly format.
        fold_scales (list[int] | None): Optional scales to evaluate per sample. When omitted,
            ``[1]`` is used for ``nocontext`` and ``[1, 3, 5]`` is used for ``global_folding``.
        two_qubit_error_rates (np.ndarray | list[float] | None): Optional array of two-qubit error rates to use.

    Raises:
        ValueError: If `samples_per_nm` exceeds `num_samples`.

    Returns:
        list[tuple[np.ndarray, tuple[float, ...]]]: The generated ideal/noisy expectation values.
    """
    if samples_per_nm > num_samples:
        raise ValueError("Samples per noise model cannot exceed number of samples.")
    if context not in {'nocontext', 'global_folding'}:
        raise ValueError("Context must be either 'nocontext' or 'global_folding'.")

    if fold_scales is None:
        fold_scales = [1, 3, 5] if context == 'global_folding' else [1]
    else:
        fold_scales = list(fold_scales)
    if not fold_scales:
        raise ValueError("fold_scales must not be empty.")

    if two_qubit_error_rates is None:
        two_qubit_error_rates = np.logspace(-3, -1)
    else:
        two_qubit_error_rates = np.asarray(two_qubit_error_rates, dtype=np.float64).reshape(-1)

    if generation_config is None:
        generation_config = DatasetGenerationConfig()
    if generation_config.use_noise_injection and generation_config.noise_injection_cluster_size <= 0:
        raise ValueError("noise_injection_cluster_size must be positive when noise injection is enabled.")

    if two_qubit_error_rates.size == 0:
        raise ValueError("two_qubit_error_rates must not be empty.")
    if np.any(two_qubit_error_rates <= 0):
        raise ValueError("two_qubit_error_rates must contain only positive values.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    nqubits = 4
    obs = SparsePauliOp(nqubits * 'Z')
    data = []
    sample_error_rates = []
    sample_single_qubit_error_rates = []
    sample_noise_model_ids = []

    num_noise_models = max(1, num_samples // samples_per_nm)
    tq_err_rates = np.resize(two_qubit_error_rates, num_noise_models)
    sq_err_rates = tq_err_rates * 0.1

    noise_models = [
        _generate_noise_model(tq_err_rate, sq_err_rate)
        for tq_err_rate, sq_err_rate in zip(tq_err_rates, sq_err_rates)
    ]
    estimator = Estimator(mode=AerSimulator())
    estimators_noisy = [
        Estimator(mode=AerSimulator(noise_model=nm, method="density_matrix"))
        for nm in noise_models
    ]
    simulators_noisy = [
        AerSimulator(noise_model=nm, method="density_matrix")
        if generation_config.use_noise_injection else None
        for nm in noise_models
    ]

    circuit = n_local(nqubits, 'ry', 'cx', reps=10, entanglement='linear')

    samples_per_nm = num_samples // num_noise_models
    for model_idx, estimator_noisy in enumerate(tqdm(estimators_noisy, disable=not progress)):
        tq_err_rate = tq_err_rates[model_idx]
        sq_err_rate = sq_err_rates[model_idx]
        simulator_noisy = simulators_noisy[model_idx]
        for _ in range(samples_per_nm):
            rotations = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)
            ideal_exp_val = estimator.run([
                (circuit.decompose(reps=1), obs, rotations)
            ]).result()[0].data.evs

            noisy_exp_vals = []
            for scale in fold_scales:
                circuit_for_scale = circuit.decompose(reps=1)
                if scale > 1:
                    circuit_for_scale = PassManager([
                        GlobalFoldingPass(scale=scale)
                    ]).run(circuit_for_scale)

                noisy_exp_val = estimator_noisy.run([
                    (circuit_for_scale, obs, rotations)
                ]).result()[0].data.evs
                if generation_config.use_noise_injection:
                    noisy_exp_val = _sample_observable_values(noisy_exp_val, generation_config)
                noisy_exp_vals.append(noisy_exp_val)

            data.append((ideal_exp_val, tuple(noisy_exp_vals)))
            sample_error_rates.append(float(tq_err_rate))
            sample_single_qubit_error_rates.append(float(sq_err_rate))
            sample_noise_model_ids.append(model_idx)

    if output_file is not None:
        save_path = Path(output_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        inputs, targets, metadata = _build_dataset_tensors(
            data,
            nqubits=nqubits,
            seed=seed,
            error_rates=two_qubit_error_rates,
            sample_error_rates=sample_error_rates,
            sample_single_qubit_error_rates=sample_single_qubit_error_rates,
            sample_noise_model_ids=sample_noise_model_ids,
        )

        metadata["noise_injection_generation"] = bool(getattr(generation_config, "use_noise_injection", False))
        if getattr(generation_config, "use_noise_injection", False):
            metadata["noise_injection_cluster_size"] = int(generation_config.noise_injection_cluster_size)
            metadata["noise_injection_noise_type"] = generation_config.noise_injection_noise_type
            metadata["noise_injection_noise_amount"] = float(generation_config.noise_injection_noise_amount)
            metadata["noise_injection_noise_distribution"] = generation_config.noise_injection_noise_distribution

        payload = {
            "inputs": inputs,
            "targets": targets,
            "metadata": metadata,
        }
        if sample_error_rates:
            payload["sample_error_rates"] = torch.tensor(sample_error_rates, dtype=torch.float32)
        if sample_single_qubit_error_rates:
            payload["sample_single_qubit_error_rates"] = torch.tensor(sample_single_qubit_error_rates, dtype=torch.float32)
        if sample_noise_model_ids:
            payload["sample_noise_model_ids"] = torch.tensor(sample_noise_model_ids, dtype=torch.int64)

        torch.save(payload, save_path)
        print(f"Saved dataset to {save_path}")

    return data


# ============================================================================
# NNAS EXTENSION -- everything below is new.
# ============================================================================

def real_amplitudes_layer_rate(p_single: float, p_two: float, n_qubits: int,
                                is_first_layer: bool = False) -> float:
    """
    Effective per-layer noise rate (the "prior" NNAS needs), computed the
    same way as qem_dataset.layer_effective_rate: p_layer = 1 - prod_d(1-p^d)
    over the gates in one repetition of the ansatz:
      - n_qubits single-qubit Ry gates (the first layer folds in the
        initial Ry block too, so it carries 2*n_qubits instead of n_qubits)
      - (n_qubits - 1) CX gates (linear entanglement)
    Uses the sample's own single/two-qubit error rates directly (no T1/T2
    derivation needed, since this dataset's noise is plain depolarizing
    noise with directly-specified probabilities).
    """
    n_single_gates = n_qubits * (2 if is_first_layer else 1)
    n_two_qubit_gates = n_qubits - 1
    survival = (1 - p_single) ** n_single_gates * (1 - p_two) ** n_two_qubit_gates
    return 1.0 - survival


def real_amplitudes_prior_sequence(p_single: float, p_two: float,
                                    n_qubits: int, n_layers: int) -> np.ndarray:
    """p_hat_l for l = 1..n_layers (first layer folds in the initial Ry block)."""
    return np.array([
        real_amplitudes_layer_rate(p_single, p_two, n_qubits, is_first_layer=(l == 0))
        for l in range(n_layers)
    ])


@dataclass
class RASequence:
    """Per-layer sequence for one RealAmplitudes sample -- the RealAmplitudes
    analogue of qem_dataset.Sequence, exposing the same (L, y_noiseless,
    y_noisy, p_hat, spec_features) interface so the two circuit families can
    share a single trainer."""
    n_qubits: int
    p_single: float
    p_two: float
    noise_model_id: int
    L: int
    y_noiseless: np.ndarray  # (L,)
    y_noisy: np.ndarray      # (L,)
    p_hat: np.ndarray        # (L,)

    def spec_features(self, L_max: int) -> np.ndarray:
        """Per-layer features: [normalized layer index, single-qubit
        error rate, two-qubit error rate] (constant across layers except
        the index) -- the RealAmplitudes analogue of
        qem_dataset.Sequence.spec_features."""
        layer_idx = np.arange(1, self.L + 1) / L_max
        p1 = np.full(self.L, self.p_single)
        p2 = np.full(self.L, self.p_two)
        return np.stack([layer_idx, p1, p2], axis=-1)  # (L, 3)


def generate_layerwise_data(
    num_samples: int,
    samples_per_nm: int,
    nqubits: int = 4,
    n_layers: int = 10,
    progress: bool = True,
    seed: int = 42,
    output_file="qem_layerwise_dataset.pt",
    two_qubit_error_rates=None,
    generation_config: DatasetGenerationConfig = None,
):
    """
    Per-layer counterpart of `generate_data`, built to be BACKWARD
    COMPATIBLE with its saved format: the same ansatz
    (n_local(nqubits, 'ry', 'cx', reps=n_layers, entanglement='linear')),
    the same `_generate_noise_model` depolarizing noise, the same
    num_noise_models / samples_per_nm scheme, and -- critically -- the same
    RNG consumption pattern (one np.random.uniform(-pi, pi, n_params) draw
    per sample, same order), so with matching (seed, num_samples,
    samples_per_nm, nqubits, two_qubit_error_rates) and n_layers==reps, the
    rotations drawn here are bit-identical to those `generate_data` would
    draw, and the final-layer values agree with what `generate_data(...,
    context='nocontext')` would compute for the same full circuit.

    The saved file's core -- "inputs", "targets", "metadata",
    "sample_error_rates", "sample_single_qubit_error_rates",
    "sample_noise_model_ids" -- is built by calling `_build_dataset_tensors`
    (unmodified) on just the FINAL layer's (ideal, noisy) values, i.e.
    exactly what `generate_data(context='nocontext', ...)` would produce:
    any existing code that loads a `generate_data` file by reading those
    keys can load a `generate_layerwise_data` file unchanged.

    Two NEW keys are added on top (ignored by code that doesn't know about
    them): "layerwise_ideal_targets" and "layerwise_noisy_inputs", each of
    shape (num_samples, n_layers) -- the full per-layer sequences that NNAS
    needs. ZNE folding is ignored per the task spec: only the scale=1
    (unfolded) noisy circuit is simulated at each layer.
    """
    if samples_per_nm > num_samples:
        raise ValueError("Samples per noise model cannot exceed number of samples.")

    if two_qubit_error_rates is None:
        two_qubit_error_rates = np.logspace(-3, -1)
    else:
        two_qubit_error_rates = np.asarray(two_qubit_error_rates, dtype=np.float64).reshape(-1)
    if two_qubit_error_rates.size == 0 or np.any(two_qubit_error_rates <= 0):
        raise ValueError("two_qubit_error_rates must be a non-empty array of positive values.")

    if generation_config is None:
        generation_config = DatasetGenerationConfig()
    if generation_config.use_noise_injection and generation_config.noise_injection_cluster_size <= 0:
        raise ValueError("noise_injection_cluster_size must be positive when noise injection is enabled.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = SparsePauliOp(nqubits * 'Z')

    num_noise_models = max(1, num_samples // samples_per_nm)
    tq_err_rates = np.resize(two_qubit_error_rates, num_noise_models)
    sq_err_rates = tq_err_rates * 0.1  # same "derived" convention as generate_data

    noise_models = [
        _generate_noise_model(tq, sq) for tq, sq in zip(tq_err_rates, sq_err_rates)
    ]
    estimator_ideal = Estimator(mode=AerSimulator())
    estimators_noisy = [
        Estimator(mode=AerSimulator(noise_model=nm, method="density_matrix"))
        for nm in noise_models
    ]

    # Pre-build the l=1..n_layers sub-circuits once (shared across samples;
    # only the bound parameter values change per sample). sub_circuits[-1]
    # is exactly what generate_data's `circuit.decompose(reps=1)` would be
    # for reps=n_layers.
    sub_circuits = [
        n_local(nqubits, 'ry', 'cx', reps=l, entanglement='linear').decompose(reps=1)
        for l in range(1, n_layers + 1)
    ]
    full_num_params = nqubits * (n_layers + 1)

    ideal_seqs, noisy_seqs = [], []
    sample_error_rates = []              # two-qubit rate (matches generate_data's variable name)
    sample_single_qubit_error_rates = []
    sample_noise_model_ids = []

    samples_per_nm = num_samples // num_noise_models
    for model_idx, estimator_noisy in enumerate(tqdm(estimators_noisy, disable=not progress)):
        tq_err_rate = tq_err_rates[model_idx]
        sq_err_rate = sq_err_rates[model_idx]
        for _ in range(samples_per_nm):
            # Same single RNG draw, same size, same order as generate_data's
            # `rotations = np.random.uniform(-np.pi, np.pi, circuit.num_parameters)`
            rotations = np.random.uniform(-np.pi, np.pi, full_num_params)

            pubs = [(sub, obs, rotations[:sub.num_parameters]) for sub in sub_circuits]
            ideal_result = estimator_ideal.run(pubs).result()
            noisy_result = estimator_noisy.run(pubs).result()

            ideal_values = [float(ideal_result[i].data.evs) for i in range(n_layers)]
            noisy_values = [float(noisy_result[i].data.evs) for i in range(n_layers)]

            ideal_seqs.append(ideal_values)
            noisy_seqs.append(noisy_values)
            sample_error_rates.append(float(tq_err_rate))
            sample_single_qubit_error_rates.append(float(sq_err_rate))
            sample_noise_model_ids.append(model_idx)

    # Backward-compatible core: reuse _build_dataset_tensors UNMODIFIED on
    # just the final layer's (ideal, noisy) pair, i.e. exactly the
    # context='nocontext' shape/semantics generate_data would produce.
    # If noise injection is enabled, apply it to the final-layer noisy
    # value exactly as generate_data does per fold -- note this means
    # "inputs" may then differ from "layerwise_noisy_inputs[:, -1]", which
    # intentionally stays as the raw physically-simulated value throughout,
    # since that's the genuine noise-accumulation signal NNAS needs;
    # noise injection is a synthetic augmentation for the other models only.
    flat_data = []
    for i in range(len(ideal_seqs)):
        final_noisy = noisy_seqs[i][-1]
        if generation_config.use_noise_injection:
            final_noisy = _sample_observable_values(final_noisy, generation_config)
        flat_data.append((ideal_seqs[i][-1], (final_noisy,)))

    inputs, targets, metadata = _build_dataset_tensors(
        flat_data,
        nqubits=nqubits,
        seed=seed,
        error_rates=two_qubit_error_rates,
        sample_error_rates=sample_error_rates,
        sample_single_qubit_error_rates=sample_single_qubit_error_rates,
        sample_noise_model_ids=sample_noise_model_ids,
    )
    metadata["n_layers"] = n_layers  # additive; harmless for readers that don't expect it
    metadata["noise_injection_generation"] = bool(generation_config.use_noise_injection)
    if generation_config.use_noise_injection:
        metadata["noise_injection_cluster_size"] = int(generation_config.noise_injection_cluster_size)
        metadata["noise_injection_noise_type"] = generation_config.noise_injection_noise_type
        metadata["noise_injection_noise_amount"] = float(generation_config.noise_injection_noise_amount)
        metadata["noise_injection_noise_distribution"] = generation_config.noise_injection_noise_distribution

    payload = {
        "inputs": inputs,      # (N, 1) -- final-layer noisy value, same as generate_data(context='nocontext')
        "targets": targets,    # (N,)   -- final-layer ideal value
        "metadata": metadata,
        # NEW, additive keys: full per-layer sequences for NNAS. Existing
        # code that only reads inputs/targets/metadata/sample_* is
        # unaffected by their presence.
        "layerwise_ideal_targets": torch.tensor(ideal_seqs, dtype=torch.float32),  # (N, n_layers)
        "layerwise_noisy_inputs": torch.tensor(noisy_seqs, dtype=torch.float32),   # (N, n_layers)
    }
    # Same top-level tensor key names as generate_data's save block
    # (note: "sample_error_rates" here, NOT "sample_two_qubit_error_rates" --
    # that longer name is only used inside `metadata`, matching
    # generate_data's own existing (slightly inconsistent) convention,
    # replicated here on purpose for exact compatibility).
    if sample_error_rates:
        payload["sample_error_rates"] = torch.tensor(sample_error_rates, dtype=torch.float32)
    if sample_single_qubit_error_rates:
        payload["sample_single_qubit_error_rates"] = torch.tensor(sample_single_qubit_error_rates, dtype=torch.float32)
    if sample_noise_model_ids:
        payload["sample_noise_model_ids"] = torch.tensor(sample_noise_model_ids, dtype=torch.int64)

    if output_file is not None:
        save_path = Path(output_file)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_path)
        print(f"Saved layer-wise dataset to {save_path} "
              f"(inputs/targets/metadata/sample_* are backward compatible with generate_data's format)")

    return payload


def _payload_to_sequences(payload: dict, nqubits: int, n_layers: int) -> list:
    """Convert a generate_layerwise_data payload (in memory or loaded from
    disk) into a list of RASequence objects."""
    if "layerwise_ideal_targets" not in payload or "layerwise_noisy_inputs" not in payload:
        raise ValueError(
            "This dataset doesn't contain per-layer sequences "
            "('layerwise_ideal_targets' / 'layerwise_noisy_inputs') -- it looks like a "
            "flat, single-final-value dataset (e.g. one produced by the original "
            "generate_data()). NNAS needs genuine per-layer supervision and can't "
            "reconstruct it from a single final value; regenerate the dataset with "
            "generate_layerwise_data(...) instead."
        )

    meta = payload.get("metadata", {})
    if "nqubits" in meta and meta["nqubits"] != nqubits:
        raise ValueError(f"nqubits mismatch: file has {meta['nqubits']}, got {nqubits}")
    if "n_layers" in meta and meta["n_layers"] != n_layers:
        raise ValueError(f"n_layers mismatch: file has {meta['n_layers']}, got {n_layers}")

    ideal = payload["layerwise_ideal_targets"].numpy()
    noisy = payload["layerwise_noisy_inputs"].numpy()
    p_two = payload["sample_error_rates"].numpy()
    p_single = payload["sample_single_qubit_error_rates"].numpy()
    model_ids = payload["sample_noise_model_ids"].numpy()

    sequences = []
    for i in range(ideal.shape[0]):
        p_hat = real_amplitudes_prior_sequence(float(p_single[i]), float(p_two[i]), nqubits, n_layers)
        sequences.append(RASequence(
            n_qubits=nqubits, p_single=float(p_single[i]), p_two=float(p_two[i]),
            noise_model_id=int(model_ids[i]), L=n_layers,
            y_noiseless=ideal[i], y_noisy=noisy[i], p_hat=p_hat,
        ))
    return sequences


def generate_real_amplitudes_dataset(
    n_sequences: int,
    n_qubits: int = 4,
    n_layers: int = 10,
    samples_per_nm: int = None,
    seed: int = 0,
    two_qubit_error_rates=None,
    output_file=None,
) -> list:
    """
    Convenience wrapper: generate n_sequences RASequence objects in-process
    via `generate_layerwise_data` (no file I/O), for quick experimentation
    or testing. For a saved, backward-compatible dataset file, use
    `generate_layerwise_data(..., output_file=...)` followed by
    `load_layerwise_dataset(...)`.
    
    Note on `two_qubit_error_rates`: `generate_layerwise_data` (like the
    original `generate_data`) picks per-noise-model rates via
    `np.resize(two_qubit_error_rates, num_noise_models)`, which -- given
    the default 50-point `np.logspace(-3, -1)` -- silently just takes the
    first few (smallest) values whenever num_noise_models is small, rather
    than spreading across the full range. To get a demo/test set that
    actually spans a meaningful range of noise strengths, this wrapper
    defaults to `np.logspace(-3, -1, num_noise_models)` instead (sized to
    match exactly, so resize is a no-op) unless the caller overrides it.
    """
    if samples_per_nm is None:
        samples_per_nm = max(1, n_sequences // 4)
    if two_qubit_error_rates is None:
        num_noise_models = max(1, n_sequences // samples_per_nm)
        two_qubit_error_rates = np.logspace(-3, -1, num_noise_models)
    payload = generate_layerwise_data(
        num_samples=n_sequences, samples_per_nm=samples_per_nm,
        nqubits=n_qubits, n_layers=n_layers, seed=seed, output_file=output_file,
        progress=True, two_qubit_error_rates=two_qubit_error_rates,
    )
    return _payload_to_sequences(payload, n_qubits, n_layers)


def load_layerwise_dataset(path, n_qubits: int, n_layers: int) -> list:
    """
    Load a layer-wise dataset saved by `generate_layerwise_data(...,
    output_file=path)` and return a list of RASequence objects ready for
    the unified NNAS trainer.

    n_qubits/n_layers describe the fixed ansatz configuration used to
    produce the file (same convention as generate_data's flat schema: not
    stored per-sample) -- if the file's own metadata carries them (as
    generate_layerwise_data's does), they're cross-checked against what's
    passed in.
    """
    payload = torch.load(path, map_location="cpu")
    return _payload_to_sequences(payload, n_qubits, n_layers)


if __name__ == "__main__":
    # Quick sanity check (small scale): generate a handful of layer-wise
    # samples in-process and print per-layer noisy vs. ideal MAE, which
    # should be non-trivial and roughly grow with layer depth, mirroring
    # qem_dataset.py's own __main__ sanity check.
    seqs = generate_real_amplitudes_dataset(n_sequences=300, n_qubits=4, n_layers=10, samples_per_nm=30, seed=45, output_file="layerwise_dataset.pt")
    # errs = np.array([np.abs(s.y_noisy - s.y_noiseless) for s in seqs])
    # print("per-layer MAE (n_qubits=4, n_layers=10):", np.round(errs.mean(axis=0), 4))
    # print("p_hat (first sample):", np.round(seqs[0].p_hat, 4))
    