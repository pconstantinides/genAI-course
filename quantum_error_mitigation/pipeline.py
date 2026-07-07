"""High-level training pipeline for the QEM sequence model."""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from qcfd.projects.qem_vqa.ml_qem.data_utils import load_dataset, make_dataloaders
from qcfd.projects.qem_vqa.ml_qem.mlp_qem import ModelConfig, SequenceExtrapolator
from qcfd.projects.qem_vqa.ml_qem.training_utils import evaluate, train_model


def run_training_pipeline(
    dataset_path: Union[str, List[str], Tuple[str, ...]],
    model_config: ModelConfig,
    train_config: object,
) -> Tuple[object, Dict, Dict[str, float]]:
    if model_config.feature_config.use_error_rate:
        inputs, targets, metadata, error_rates = load_dataset(dataset_path, return_error_rates=True)
    else:
        inputs, targets, metadata = load_dataset(dataset_path)
        error_rates = None

    metadata_summary = {
        "nqubits": metadata.get("nqubits", None),
        "num_samples": metadata.get("num_samples", inputs.shape[0]),
        "nfolds": metadata.get("nfolds", None),
    }
    if error_rates is not None:
        metadata_summary["error_rate_range"] = [float(error_rates.min().item()), float(error_rates.max().item())]
    print(f"Loaded {inputs.shape[0]} samples with input shape {tuple(inputs.shape)}. metadata={metadata_summary}")

    if inputs.ndim == 2:
        model_config.seq_len = inputs.shape[1]

    train_loader, val_loader, test_loader = make_dataloaders(
        inputs,
        targets,
        batch_size=train_config.batch_size,
        val_frac=train_config.val_frac,
        test_frac=train_config.test_frac,
        seed=train_config.seed,
        error_rates=error_rates,
    )

    if train_config.dropout is not None:
        model_config.dropout = train_config.dropout
    if train_config.dropout_rates is not None:
        model_config.dropout_rates = train_config.dropout_rates
    if train_config.weight_decay is not None:
        model_config.weight_decay = train_config.weight_decay

    model = SequenceExtrapolator(model_config)
    result = train_model(model, train_loader, val_loader, train_config)

    device = torch.device(train_config.device)
    test_metrics = evaluate(
        model,
        test_loader,
        nn.MSELoss(),
        device,
        result["input_standardizer"],
        result["target_standardizer"],
    )
    print("Test metrics:", test_metrics)

    return model, result, test_metrics
