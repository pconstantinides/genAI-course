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

from qiskit_helpers.passes.folding import GlobalFoldingPass


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
    error_rates: np.ndarray | list[float] | None = None,
    sample_error_rates: np.ndarray | list[float] | None = None,
    sample_single_qubit_error_rates: np.ndarray | list[float] | None = None,
    sample_noise_model_ids: np.ndarray | list[int] | None = None,
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
    source_dataset_path: str | Path,
    output_file: str | None,
    generation_config: DatasetGenerationConfig | None = None,
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
    output_file: str | None = "qem_dataset.pt",
    fold_scales: list[int] | None = None,
    two_qubit_error_rates: np.ndarray | list[float] | None = None,
    generation_config: DatasetGenerationConfig | None = None,
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


def visualize_data(data):
    if not data:
        return

    ideal_exp_vals = np.asarray([sample[0] for sample in data], dtype=np.float64).reshape(-1)
    noisy_samples = np.asarray([_as_feature_vector(sample[1]) for sample in data], dtype=np.float64)

    if plt is None:
        return

    if noisy_samples.ndim == 1:
        noisy_samples = noisy_samples.reshape(-1, 1)

    for idx in range(noisy_samples.shape[1]):
        plt.scatter(
            ideal_exp_vals,
            noisy_samples[:, idx],
            alpha=0.5,
            edgecolors='w',
            label=f"scale {idx + 1}",
        )

    plt.plot([-1, 1], [-1, 1], 'r-', linewidth=1)
    plt.xlabel("Ideal Expectation Value", fontweight='bold')
    plt.ylabel("Noisy Expectation Value", fontweight='bold')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = generate_data(
        75_000,
        300,
        context="global_folding",
        output_file="data/qem_dataset_v2.pt",
        two_qubit_error_rates=np.logspace(-3, -1, 75_000//300)
    )    
    hn_data = generate_data(
        35_000,
        256,
        context="global_folding",
        output_file="data/high_noise_dataset_v2.pt",
        two_qubit_error_rates=np.logspace(-1.5, -1, 35_000//256)
    )
    ln_data = generate_data(
        35_000,
        256,
        context="global_folding",
        output_file="data/low_noise_dataset_v2.pt",
        two_qubit_error_rates=np.logspace(-3, -2.5, 35_000//256)
    )
    visualize_data(hn_data)
    visualize_data(ln_data)