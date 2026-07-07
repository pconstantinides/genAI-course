"""Training, evaluation, and diagnostics helpers for the QEM sequence model."""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from qcfd.projects.qem_vqa.ml_qem.data_utils import SequenceDataset
from qcfd.projects.qem_vqa.ml_qem.mlp_qem import SequenceExtrapolator


class Standardizer:
    """Zero-mean, unit-variance scaling, fit once on training data."""

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "Standardizer":
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def fit_standardizers(loader: torch.utils.data.DataLoader) -> Tuple[Standardizer, Standardizer]:
    xs, ys = [], []
    for batch in loader:
        x, y, *_ = batch
        xs.append(x)
        ys.append(y)
    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    return Standardizer().fit(xs), Standardizer().fit(ys)


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds - targets).abs().mean().item()


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(((preds - targets) ** 2).mean()).item()


def r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-12)
    return (1 - ss_res / ss_tot).item()


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    x, y = batch[0], batch[1]
    error_rates = batch[2] if len(batch) > 2 else None
    return x, y, error_rates


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    input_standardizer: Optional[Standardizer] = None,
    target_standardizer: Optional[Standardizer] = None,
    grad_clip: Optional[float] = None,
) -> float:
    model.train()
    total_loss, n_seen = 0.0, 0

    for batch in loader:
        x, y, error_rates = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        if input_standardizer is not None:
            x = input_standardizer.transform(x)
        y_target = target_standardizer.transform(y) if target_standardizer is not None else y
        if error_rates is not None:
            error_rates = error_rates.to(device)

        optimizer.zero_grad()
        preds = model(x, error_rates=error_rates)
        loss = loss_fn(preds, y_target)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)

    return total_loss / n_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    input_standardizer: Optional[Standardizer] = None,
    target_standardizer: Optional[Standardizer] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_preds, all_targets = [], []

    for batch in loader:
        x, y, error_rates = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_in = input_standardizer.transform(x) if input_standardizer is not None else x
        y_target = target_standardizer.transform(y) if target_standardizer is not None else y
        if error_rates is not None:
            error_rates = error_rates.to(device)

        preds_norm = model(x_in, error_rates=error_rates)
        loss = loss_fn(preds_norm, y_target)
        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)

        preds = (
            target_standardizer.inverse_transform(preds_norm)
            if target_standardizer is not None
            else preds_norm
        )
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)

    return {
        "loss": total_loss / n_seen,
        "mae": mae(preds_cat, targets_cat),
        "rmse": rmse(preds_cat, targets_cat),
        "r2": r2_score(preds_cat, targets_cat),
    }


@torch.no_grad()
def evaluate_by_group(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
    device: torch.device,
    input_standardizer: Optional[Standardizer] = None,
    target_standardizer: Optional[Standardizer] = None,
    error_rates: Optional[torch.Tensor] = None,
) -> Dict[float, Dict[str, float]]:
    model.eval()
    results = {}
    for g in torch.unique(groups):
        mask = groups == g
        x = inputs[mask].to(device)
        y = targets[mask].to(device)
        group_error_rates = error_rates[mask].to(device) if error_rates is not None else None
        x_in = input_standardizer.transform(x) if input_standardizer is not None else x
        preds_norm = model(x_in, error_rates=group_error_rates)
        preds = (
            target_standardizer.inverse_transform(preds_norm)
            if target_standardizer is not None
            else preds_norm
        )
        results[float(g.item())] = {
            "n": int(mask.sum().item()),
            "mae": mae(preds.cpu(), y.cpu()),
            "rmse": rmse(preds.cpu(), y.cpu()),
            "r2": r2_score(preds.cpu(), y.cpu()),
        }
    return results


@torch.no_grad()
def diagnose_outliers(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    input_standardizer: Optional[Standardizer] = None,
    target_standardizer: Optional[Standardizer] = None,
    outlier_method: str = "iqr",
    threshold: float = 1.5,
) -> Dict[str, object]:
    model.eval()
    all_preds, all_targets, all_inputs = [], [], []

    for batch in loader:
        x, y, batch_error_rates = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        x_in = input_standardizer.transform(x) if input_standardizer is not None else x
        if batch_error_rates is not None:
            batch_error_rates = batch_error_rates.to(device)
        preds_norm = model(x_in, error_rates=batch_error_rates)
        preds = (
            target_standardizer.inverse_transform(preds_norm)
            if target_standardizer is not None
            else preds_norm
        )
        all_inputs.append(x.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    inputs_cat = torch.cat(all_inputs, dim=0)
    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    abs_errors = (preds_cat - targets_cat).abs()
    rel_errors = abs_errors / (targets_cat.abs() + 1e-8)

    if outlier_method == "iqr":
        q1 = torch.quantile(abs_errors, 0.25)
        q3 = torch.quantile(abs_errors, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask = (abs_errors < lower_bound) | (abs_errors > upper_bound)
    elif outlier_method == "std":
        mean_err = abs_errors.mean()
        std_err = abs_errors.std()
        lower_bound = mean_err - threshold * std_err
        upper_bound = mean_err + threshold * std_err
        outlier_mask = (abs_errors < lower_bound) | (abs_errors > upper_bound)
    else:
        raise ValueError(f"Unknown outlier_method: {outlier_method}")

    outlier_indices = torch.where(outlier_mask)[0]
    outlier_data = []
    for idx in outlier_indices:
        outlier_data.append({
            "input": inputs_cat[idx].numpy(),
            "prediction": preds_cat[idx].item(),
            "target": targets_cat[idx].item(),
            "absolute_error": abs_errors[idx].item(),
            "relative_error": rel_errors[idx].item(),
        })

    outlier_data.sort(key=lambda x: x["absolute_error"], reverse=True)

    return {
        "n_total": inputs_cat.shape[0],
        "n_outliers": len(outlier_indices),
        "outlier_fraction": float(len(outlier_indices) / inputs_cat.shape[0]),
        "outlier_indices": outlier_indices.numpy(),
        "outlier_data": outlier_data,
        "error_stats": {
            "mean_abs_error": abs_errors.mean().item(),
            "std_abs_error": abs_errors.std().item(),
            "min_abs_error": abs_errors.min().item(),
            "max_abs_error": abs_errors.max().item(),
            "median_abs_error": abs_errors.median().item(),
        },
    }


def print_outlier_diagnostics(outlier_results: Dict, max_display: int = 10) -> None:
    print("\n" + "=" * 80)
    print("OUTLIER DIAGNOSIS REPORT")
    print("=" * 80)

    print(f"\nTotal samples: {outlier_results['n_total']}")
    print(f"Outliers detected: {outlier_results['n_outliers']} "
          f"({outlier_results['outlier_fraction'] * 100:.2f}%)")

    print("\nError Statistics:")
    stats = outlier_results["error_stats"]
    print(f"  Mean absolute error:   {stats['mean_abs_error']:.6e}")
    print(f"  Std absolute error:    {stats['std_abs_error']:.6e}")
    print(f"  Min absolute error:    {stats['min_abs_error']:.6e}")
    print(f"  Max absolute error:    {stats['max_abs_error']:.6e}")
    print(f"  Median absolute error: {stats['median_abs_error']:.6e}")

    if outlier_results["n_outliers"] > 0:
        print(f"\n{'-' * 80}")
        print(f"Top {min(max_display, len(outlier_results['outlier_data']))} Outliers (sorted by error):")
        print(f"{'-' * 80}")

        for i, outlier in enumerate(outlier_results["outlier_data"][:max_display], 1):
            print(f"\nOutlier #{i}:")
            print(f"  Input sequence:   {outlier['input']}")
            print(f"  Prediction:       {outlier['prediction']:.6e}")
            print(f"  Ground truth:     {outlier['target']:.6e}")
            print(f"  Absolute error:   {outlier['absolute_error']:.6e}")
            print(f"  Relative error:   {outlier['relative_error']:.6e}")
    else:
        print("\nNo outliers detected!")

    print("\n" + "=" * 80 + "\n")


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: object,
) -> Dict:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    model.to(device)

    input_standardizer = target_standardizer = None
    if config.normalize:
        input_standardizer, target_standardizer = fit_standardizers(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_scheduler_factor, patience=config.lr_scheduler_patience
    )
    loss_fn = config.objective

    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_r2": [], "lr": []}
    best_val_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            input_standardizer, target_standardizer, config.grad_clip,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, input_standardizer, target_standardizer)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_r2"].append(val_metrics["r2"])
        history["lr"].append(current_lr)

        improved = val_metrics["loss"] < best_val_loss - config.min_delta
        if improved:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config.log_every and epoch % config.log_every == 0:
            print(
                f"[epoch {epoch:4d}] train_loss={train_loss:.6f} "
                f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f} "
                f"val_r2={val_metrics['r2']:.4f} lr={current_lr:.2e}"
            )

        if epochs_no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {config.patience} epochs).")
            break

    model.load_state_dict(best_state)
    if config.checkpoint_path:
        save_checkpoint(model, input_standardizer, target_standardizer, config.checkpoint_path)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "input_standardizer": input_standardizer,
        "target_standardizer": target_standardizer,
    }


def save_checkpoint(
    model: SequenceExtrapolator,
    input_standardizer: Optional[Standardizer],
    target_standardizer: Optional[Standardizer],
    path: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "input_standardizer": input_standardizer,
            "target_standardizer": target_standardizer,
        },
        path,
    )


def load_checkpoint(
    path: str, device: str = "cpu"
) -> Tuple[SequenceExtrapolator, Optional[Standardizer], Optional[Standardizer]]:
    blob = torch.load(path, map_location=device, weights_only=False)
    model = SequenceExtrapolator(blob["model_config"])
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, blob.get("input_standardizer"), blob.get("target_standardizer")
