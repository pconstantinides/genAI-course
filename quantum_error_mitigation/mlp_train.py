"""Compatibility wrapper for the split QEM training modules."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from qcfd.projects.qem_vqa.ml_qem.data_utils import SequenceDataset, load_dataset, make_dataloaders
from qcfd.projects.qem_vqa.ml_qem.mlp_qem import ModelConfig, SequenceExtrapolator
from qcfd.projects.qem_vqa.ml_qem.pipeline import run_training_pipeline
from qcfd.projects.qem_vqa.ml_qem.training_utils import (
    Standardizer,
    diagnose_outliers,
    evaluate,
    fit_standardizers,
    load_checkpoint,
    print_outlier_diagnostics,
    save_checkpoint,
    train_model,
    train_one_epoch,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    epochs: int = 200
    objective: Optional[nn.Module] = nn.MSELoss()
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: Optional[float] = None
    dropout_rates: Optional[List[float]] = None
    batch_size: int = 64
    val_frac: float = 0.15
    test_frac: float = 0.15
    normalize: bool = True
    grad_clip: Optional[float] = 1.0
    patience: int = 20
    min_delta: float = 1e-6
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: Optional[str] = None
    log_every: int = 10


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
    device: str = "cpu",
    input_standardizer: Optional[Standardizer] = None,
    target_standardizer: Optional[Standardizer] = None,
    error_rates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    model.eval()
    x = x.float()
    if x.dim() == 1:
        seq_len = getattr(getattr(model, "config", None), "seq_len", None)
        if seq_len == 1:
            x = x.unsqueeze(1)
        else:
            x = x.unsqueeze(0)

    x = x.to(device)
    if input_standardizer is not None:
        x = input_standardizer.transform(x)
    if error_rates is not None:
        error_rates = error_rates.to(device)
    preds = model(x, error_rates=error_rates)
    if target_standardizer is not None:
        preds = target_standardizer.inverse_transform(preds)
    return preds.cpu()


__all__ = [
    "ModelConfig",
    "SequenceExtrapolator",
    "SequenceDataset",
    "Standardizer",
    "TrainConfig",
    "diagnose_outliers",
    "evaluate",
    "fit_standardizers",
    "load_checkpoint",
    "load_dataset",
    "make_dataloaders",
    "predict",
    "print_outlier_diagnostics",
    "run_training_pipeline",
    "save_checkpoint",
    "set_seed",
    "train_model",
    "train_one_epoch",
]


if __name__ == "__main__":
    from qcfd.projects.qem_vqa.ml_qem.mlp_qem import FeatureConfig

    dataset_path = "/home/pconstant/Dev/angelakis_research_group/qcfd/projects/qem_vqa/ml_qem/data/low_noise_dataset_v2.pt"
    inputs, targets, metadata, error_rates = load_dataset(dataset_path, return_error_rates=True)

    feature_config = FeatureConfig(
        use_raw=True,
        use_error_rate=True,
        use_differences=False,
        use_ratios=True,
        use_frequency_mod=False,
        learnable_frequencies=False,
    )

    train_config = TrainConfig(
        epochs=100,
        objective=nn.HuberLoss(),
        patience=10,
        log_every=10,
        checkpoint_path="/home/pconstant/Dev/angelakis_research_group/qcfd/projects/qem_vqa/ml_qem/checkpoints/best_model.pt",
    )

    model_config = ModelConfig(
        hidden_dims=[32, 16, 8],
        activation="gelu",
        feature_config=feature_config,
        use_batch_norm=False,
        seq_len=inputs.shape[1],
    )

    model, result, test_metrics = run_training_pipeline(dataset_path, model_config, train_config)

    diagnoses = diagnose_outliers(
        model,
        torch.utils.data.DataLoader(SequenceDataset(inputs, targets, error_rates), batch_size=64),
        device=train_config.device,
        input_standardizer=result["input_standardizer"],
        target_standardizer=result["target_standardizer"],
        outlier_method="iqr",
        threshold=1.5
    )
    print_outlier_diagnostics(diagnoses, max_display=10)
