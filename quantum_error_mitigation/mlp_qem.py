"""
Sequence Extrapolator
=====================

A small configurable MLP for estimating the ideal expectation value from a
vector of noisy zero-noise extrapolation (ZNE) measurements,
for example (noisy1, noisy2, ..., noisyk) -> ideal.

Feature engineering is modular and toggled via FeatureConfig:
  - raw noisy values
  - consecutive differences
  - consecutive ratios
  - frequency-modulated (sin/cos) features

Usage:
    feat_cfg = FeatureConfig(use_raw=True, use_differences=True, use_ratios=True)
    model_cfg = ModelConfig(hidden_dims=[64, 64], feature_config=feat_cfg)
    model = SequenceExtrapolator(model_cfg)
    y_pred = model(x)   # x: (batch, k) where k is the number of noisy measurements
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    use_raw: bool = True
    use_differences: bool = True
    use_ratios: bool = False
    use_frequency_mod: bool = False
    use_error_rate: bool = False

    # Ratio settings. Sequences converging to zero can make raw ratios
    # explode/blow up near the small-value end, so the denominator is
    # floor-clamped in absolute value before dividing.
    ratio_eps: float = 1e-3

    # Frequency settings. Each input value is multiplied by each frequency
    # and passed through sin/cos, producing 2 * seq_len * n_frequencies features.
    frequencies: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    learnable_frequencies: bool = False


class FeatureEngineer(nn.Module):
    """Maps a raw input sequence (batch, seq_len) to an engineered feature
    vector (batch, feature_dim), based on the toggles in FeatureConfig."""

    def __init__(self, config: FeatureConfig, seq_len: int = 3):
        super().__init__()
        self.config = config
        self.seq_len = seq_len

        if config.use_frequency_mod:
            freq_tensor = torch.tensor(config.frequencies, dtype=torch.float32)
            if config.learnable_frequencies:
                self.frequencies = nn.Parameter(freq_tensor)
            else:
                self.register_buffer("frequencies", freq_tensor)

    def forward(self, x: torch.Tensor, error_rates: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len)
        cfg = self.config
        feats = []

        if cfg.use_raw:
            feats.append(x)

        if cfg.use_differences:
            diffs = x[:, 1:] - x[:, :-1]  # (batch, seq_len - 1)
            feats.append(diffs)

        if cfg.use_ratios:
            numerator = x[:, 1:]
            denominator = x[:, :-1]
            # Floor the magnitude of the denominator away from zero while
            # preserving its sign, to avoid blow-ups near the converging tail.
            safe_denom = torch.sign(denominator) * torch.clamp(
                denominator.abs(), min=cfg.ratio_eps
            )
            # Replace any exact-zero signs (sign(0) = 0) with +eps.
            safe_denom = torch.where(
                safe_denom == 0, torch.full_like(safe_denom, cfg.ratio_eps), safe_denom
            )
            ratios = numerator / safe_denom
            feats.append(ratios)

        if cfg.use_frequency_mod:
            # (batch, seq_len, 1) * (1, 1, n_freq) -> (batch, seq_len, n_freq)
            mod = x.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
            feats.append(torch.sin(mod).flatten(start_dim=1))
            feats.append(torch.cos(mod).flatten(start_dim=1))

        if cfg.use_error_rate:
            if error_rates is None:
                raise ValueError("error_rates must be provided when use_error_rate=True.")
            error_rates = error_rates.to(device=x.device, dtype=x.dtype).reshape(-1, 1)
            feats.append(error_rates)

        if not feats:
            raise ValueError("At least one feature type must be enabled in FeatureConfig.")

        return torch.cat(feats, dim=-1)

    @property
    def output_dim(self) -> int:
        cfg = self.config
        dim = 0
        if cfg.use_raw:
            dim += self.seq_len
        if cfg.use_differences:
            dim += self.seq_len - 1
        if cfg.use_ratios:
            dim += self.seq_len - 1
        if cfg.use_frequency_mod:
            dim += 2 * self.seq_len * len(cfg.frequencies)
        if cfg.use_error_rate:
            dim += 1
        return dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


@dataclass
class ModelConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "gelu"
    dropout: float = 0.0
    dropout_rates: Optional[List[float]] = None
    weight_decay: float = 0.0
    seq_len: int = 3
    bounded_output: bool = True  # squashes output to (-1, 1) via tanh
    use_batch_norm: bool = False  # apply batch normalization after linear layers
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


class SequenceExtrapolator(nn.Module):
    """Feature engineering + MLP regressor for ZNE-style inputs.

    The input is expected to be a batch of vectors of noisy measurements,
    shaped as (batch_size, k), where k is the number of noisy values per sample.
    The model predicts the corresponding ideal value.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{config.activation}'. "
                              f"Choose from {list(_ACTIVATIONS)}.")

        self.config = config  # stored so checkpoints can rebuild the architecture
        self.feature_engineer = FeatureEngineer(config.feature_config, seq_len=config.seq_len)
        act_cls = _ACTIVATIONS[config.activation]

        layers = []
        prev_dim = self.feature_engineer.output_dim
        for layer_idx, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls())
            if config.dropout_rates is not None and layer_idx < len(config.dropout_rates):
                layer_dropout = config.dropout_rates[layer_idx]
            else:
                layer_dropout = config.dropout
            if layer_dropout > 0:
                layers.append(nn.Dropout(layer_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        if config.bounded_output:
            layers.append(nn.Tanh())  # constrains output to (-1, 1)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, error_rates: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, seq_len) -> (batch,)
        feats = self.feature_engineer(x, error_rates=error_rates)
        return self.mlp(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    # Toggle whichever engineered features you want here.
    feature_config = FeatureConfig(
        use_raw=True,
        use_differences=True,
        use_ratios=True,
        use_frequency_mod=True,
        frequencies=[1.0, 2.0, 4.0],
    )
    model_config = ModelConfig(
        hidden_dims=[64, 64],
        activation="gelu",
        dropout=0.1,
        seq_len=3,
        use_batch_norm=True,
        feature_config=feature_config,
    )

    model = SequenceExtrapolator(model_config)

    batch_size = 8
    x = torch.linspace(0.8, 0.1, steps=3).unsqueeze(0).repeat(batch_size, 1)
    x = x + 0.01 * torch.randn_like(x)  # small noise per example

    y_pred = model(x)
    print("Input shape:", x.shape)
    print("Feature dim:", model.feature_engineer.output_dim)
    print("Output shape:", y_pred.shape)
    print("Predictions:", y_pred.detach())

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", n_params)