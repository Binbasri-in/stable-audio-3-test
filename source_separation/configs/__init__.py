"""Structured configuration dataclasses for audio models, datasets, and trainers.

Provides strongly typed configuration containers with serialization and deserialization
support, replacing sprawling parameter lists and facilitating YAML/JSON configuration files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class BaseConfig:
    """Base configuration supporting dictionary export and construction."""

    def to_dict(self) -> dict[str, Any]:
        """Converts configuration to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        """Constructs configuration instance from dictionary, filtering unknown keys."""
        valid_keys = set(cls.__annotations__)
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class AudioDataConfig(BaseConfig):
    """Audio data pipeline configuration."""

    sample_rate: int = 44100
    channels: int = 2
    segment_seconds: float = 6.0
    random_crop: bool = True
    target_stem: str = "vocals"
    silence_threshold_db: float = -45.0


@dataclass
class TrainerConfig(BaseConfig):
    """Training hyperparameter and execution configuration."""

    epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    use_amp: bool = False
    grad_clip: float = 1.0
    batch_size: int = 8
    num_workers: int = 2
    val_every: int = 1
    checkpoint_dir: str | None = "checkpoints"


@dataclass
class AutoencoderConfig(BaseConfig):
    """Waveform Convolutional Autoencoder configuration."""

    in_channels: int = 2
    out_channels: int | None = None
    base_channels: int = 32
    latent_dim: int = 128
    strides: tuple[int, ...] = (4, 4, 4, 4)
    max_channels: int = 256


@dataclass
class VAEConfig(BaseConfig):
    """Waveform Variational Autoencoder configuration."""

    in_channels: int = 2
    base_channels: int = 32
    latent_dim: int = 128
    strides: tuple[int, ...] = (4, 4, 4, 4)
    beta: float = 4.0
    capacity: float = 0.0


@dataclass
class GlowConfig(BaseConfig):
    """Waveform Invertible Normalizing Flow (Glow) configuration."""

    in_channels: int = 2
    squeeze_factor: int = 16
    num_steps: int = 8
    hidden_dim: int = 64
    num_layers_per_step: int = 4
    scale_bound: float = 3.0
    sample_rate: int = 44100


@dataclass
class JEPAConfig(BaseConfig):
    """Audio-JEPA masked latent predictive model configuration."""

    in_channels: int = 2
    patch_size: int = 16
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mask_ratio: float = 0.6
    ema_decay: float = 0.996


__all__ = [
    "AudioDataConfig",
    "AutoencoderConfig",
    "BaseConfig",
    "GlowConfig",
    "JEPAConfig",
    "TrainerConfig",
    "VAEConfig",
]
