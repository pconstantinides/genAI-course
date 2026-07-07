"""Dataset loading and batching helpers for the QEM sequence trainer."""

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class SequenceDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, error_rates: Optional[torch.Tensor] = None):
        inputs = inputs.float()
        targets = targets.float().view(-1)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same length")

        self.inputs = inputs
        self.targets = targets
        self.error_rates = None
        if error_rates is not None:
            error_rates = error_rates.float().view(-1)
            if error_rates.shape[0] != inputs.shape[0]:
                raise ValueError("error_rates must have the same number of entries as inputs")
            self.error_rates = error_rates

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if self.error_rates is None:
            return self.inputs[idx], self.targets[idx]
        return self.inputs[idx], self.targets[idx], self.error_rates[idx]


def _build_metadata_summary(metadata_list: List[dict]) -> dict:
    summary = {
        "num_datasets": len(metadata_list),
        "datasets": metadata_list,
        "nqubits": [meta.get("nqubits", -1) for meta in metadata_list],
    }

    for key in ("nfolds", "seed", "num_samples"):
        values = [meta.get(key) for meta in metadata_list if key in meta]
        if not values:
            continue
        if all(value == values[0] for value in values):
            summary[key] = values[0]
        else:
            summary[key] = values

    return summary


def _resolve_error_rates(blob: dict, metadata: dict, inputs: torch.Tensor) -> Optional[torch.Tensor]:
    error_rates = blob.get("sample_error_rates")
    if error_rates is None:
        error_rates = blob.get("sample_two_qubit_error_rates")
    if error_rates is None:
        error_rates = blob.get("error_rate")
    if error_rates is None:
        error_rates = metadata.get("sample_two_qubit_error_rates")
    if error_rates is None:
        error_rates = metadata.get("error_rate")

    if error_rates is None:
        return None

    if not torch.is_tensor(error_rates):
        error_rates = torch.as_tensor(error_rates)

    error_rates = error_rates.float().view(-1)
    if error_rates.shape[0] != inputs.shape[0]:
        raise ValueError("error_rates must have the same number of entries as inputs")
    return error_rates


def load_dataset(
    path: Union[str, List[str], Tuple[str, ...]],
    return_error_rates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, dict] | Tuple[torch.Tensor, torch.Tensor, dict, Optional[torch.Tensor]]:
    """Load one dataset file or concatenate a list of dataset files."""
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("dataset path list cannot be empty")

        if return_error_rates:
            inputs, targets, _, metadata_list, error_rates = load_multiple_datasets(list(path), return_error_rates=True)
            return inputs, targets, _build_metadata_summary(metadata_list), error_rates

        inputs, targets, _, metadata_list = load_multiple_datasets(list(path))
        return inputs, targets, _build_metadata_summary(metadata_list)

    blob = torch.load(path, map_location="cpu")
    inputs = blob["inputs"]
    targets = blob["targets"]
    metadata = blob.get("metadata", {})

    if not torch.is_tensor(inputs):
        inputs = torch.as_tensor(inputs)
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)

    inputs = inputs.float()
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(1)

    error_rates = _resolve_error_rates(blob, metadata, inputs) if return_error_rates else None

    if return_error_rates:
        return inputs, targets.float().view(-1), metadata, error_rates
    return inputs, targets.float().view(-1), metadata


def load_multiple_datasets(
    paths: List[str],
    return_error_rates: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict], torch.Tensor]:
    """Load and concatenate several dataset files."""
    all_inputs, all_targets, all_nqubits, metadatas = [], [], [], []
    all_error_rates = []
    for path in paths:
        if return_error_rates:
            inputs, targets, metadata, error_rates = load_dataset(path, return_error_rates=True)
            if error_rates is None:
                raise ValueError(f"No error rates found in dataset: {path}")
            all_error_rates.append(error_rates)
        else:
            inputs, targets, metadata = load_dataset(path)
        all_inputs.append(inputs)
        all_targets.append(targets)
        nq = float(metadata.get("nqubits", -1))
        all_nqubits.append(torch.full((inputs.shape[0],), nq))
        metadatas.append(metadata)

    if return_error_rates:
        return (
            torch.cat(all_inputs, dim=0),
            torch.cat(all_targets, dim=0),
            torch.cat(all_nqubits, dim=0),
            metadatas,
            torch.cat(all_error_rates, dim=0),
        )

    return (
        torch.cat(all_inputs, dim=0),
        torch.cat(all_targets, dim=0),
        torch.cat(all_nqubits, dim=0),
        metadatas,
    )


def make_dataloaders(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 64,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    error_rates: Optional[torch.Tensor] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split a dataset into train/val/test DataLoaders."""
    dataset = SequenceDataset(inputs, targets, error_rates=error_rates)
    n = len(dataset)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("val_frac + test_frac leaves no training examples")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=generator)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )
