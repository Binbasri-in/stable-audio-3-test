"""Model factory registry and preset constructors for audio source separation.

Provides a unified factory interface (similar to timm and torchvision.models)
to discover, inspect, and instantiate pre-configured architectures with tested hyperparameters.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from typing import Any

from torch import nn

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a model architecture or factory function in the global registry.

    Args:
        name: Unique string identifier for the model (e.g. 'waveform_vae_small').
    """

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered in source_separation registry.")
        _MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Instantiates and returns a model from the registry by name.

    Args:
        name: Name of the registered model preset.
        **kwargs: Overriding keyword arguments passed to the constructor.

    Returns:
        Instantiated nn.Module.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model '{name}'. Available registered models: [{available}]")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models(pattern: str = "*") -> list[str]:
    """Returns a sorted list of registered model names, optionally filtered by glob pattern.

    Args:
        pattern: Unix glob pattern (e.g., 'waveform_*', '*vae*').
    """
    return sorted(m for m in _MODEL_REGISTRY if fnmatch.fnmatch(m, pattern))


# ===========================================================================
# Model Architecture Presets
# ===========================================================================


@register_model("waveform_ae_small")
def waveform_ae_small(
    in_channels: int = 2,
    base_channels: int = 32,
    latent_dim: int = 64,
    strides: tuple[int, ...] = (4, 4, 4, 4),
    **kwargs: Any,
) -> nn.Module:
    """Compact 4-stage 1D convolutional waveform autoencoder (fast training / low VRAM)."""
    from .autoencoder import WaveformAutoencoder

    return WaveformAutoencoder(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        strides=strides,
        **kwargs,
    )


@register_model("waveform_ae_base")
def waveform_ae_base(
    in_channels: int = 2,
    base_channels: int = 48,
    latent_dim: int = 128,
    strides: tuple[int, ...] = (4, 4, 4, 4),
    **kwargs: Any,
) -> nn.Module:
    """Standard 4-stage 1D convolutional waveform autoencoder."""
    from .autoencoder import WaveformAutoencoder

    return WaveformAutoencoder(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        strides=strides,
        **kwargs,
    )


@register_model("waveform_vae_small")
def waveform_vae_small(
    in_channels: int = 2,
    base_channels: int = 32,
    latent_dim: int = 64,
    strides: tuple[int, ...] = (4, 4, 4, 4),
    beta: float = 1.0,
    **kwargs: Any,
) -> nn.Module:
    """Compact 4-stage 1D waveform Variational Autoencoder."""
    from .vae import WaveformBetaVAE

    return WaveformBetaVAE(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        strides=strides,
        beta=beta,
        **kwargs,
    )


@register_model("waveform_vae_base")
def waveform_vae_base(
    in_channels: int = 2,
    base_channels: int = 48,
    latent_dim: int = 128,
    strides: tuple[int, ...] = (4, 4, 4, 4),
    beta: float = 4.0,
    **kwargs: Any,
) -> nn.Module:
    """Standard 4-stage 1D waveform Beta-VAE for disentangled acoustic representation learning."""
    from .vae import WaveformBetaVAE

    return WaveformBetaVAE(
        in_channels=in_channels,
        base_channels=base_channels,
        latent_dim=latent_dim,
        strides=strides,
        beta=beta,
        **kwargs,
    )


@register_model("spectrogram_ae")
def spectrogram_ae(
    in_channels: int = 2,
    latent_dim: int = 128,
    base_channels: int = 32,
    **kwargs: Any,
) -> nn.Module:
    """2D time-frequency spectrogram autoencoder."""
    from .spectrogram import SingleSourceAutoencoder

    return SingleSourceAutoencoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        base_channels=base_channels,
        **kwargs,
    )


@register_model("spectrogram_vae")
def spectrogram_vae(
    in_channels: int = 2,
    latent_dim: int = 128,
    base_channels: int = 32,
    **kwargs: Any,
) -> nn.Module:
    """2D time-frequency spectrogram Variational Autoencoder."""
    from .spectrogram import SingleSourceVanillaVAE

    return SingleSourceVanillaVAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        base_channels=base_channels,
        **kwargs,
    )


@register_model("waveform_glow_small")
def waveform_glow_small(
    in_channels: int = 2,
    squeeze_factor: int = 16,
    num_steps: int = 6,
    hidden_dim: int = 48,
    **kwargs: Any,
) -> nn.Module:
    """Compact invertible normalizing flow for raw audio waveforms."""
    from .glow import WaveformGlow

    return WaveformGlow(
        in_channels=in_channels,
        squeeze_factor=squeeze_factor,
        num_steps=num_steps,
        hidden_dim=hidden_dim,
        **kwargs,
    )


@register_model("waveform_glow_base")
def waveform_glow_base(
    in_channels: int = 2,
    squeeze_factor: int = 16,
    num_steps: int = 10,
    hidden_dim: int = 64,
    **kwargs: Any,
) -> nn.Module:
    """Standard invertible normalizing flow for raw audio waveforms."""
    from .glow import WaveformGlow

    return WaveformGlow(
        in_channels=in_channels,
        squeeze_factor=squeeze_factor,
        num_steps=num_steps,
        hidden_dim=hidden_dim,
        **kwargs,
    )


@register_model("latent_ebm")
def latent_ebm(
    in_channels: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 4,
    kernel_size: int = 5,
    **kwargs: Any,
) -> nn.Module:
    """Scalar Energy-Based Model operating on audio latent spaces."""
    from .ebm import ScalarLatentEBM

    return ScalarLatentEBM(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        kernel_size=kernel_size,
        **kwargs,
    )


@register_model("waveform_jem")
def waveform_jem(
    in_channels: int = 2,
    base_channels: int = 32,
    num_classes: int = 4,
    num_stages: int = 4,
    strides: tuple[int, ...] = (4, 4, 4, 4),
    **kwargs: Any,
) -> nn.Module:
    """Joint Energy Model learning simultaneous classification and generative energy."""
    from .jem import WaveformJEM

    return WaveformJEM(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        num_stages=num_stages,
        strides=strides,
        **kwargs,
    )


@register_model("audio_jepa_base")
def audio_jepa_base(
    embed_dim: int = 128,
    patch_size: int = 256,
    stride: int = 128,
    depth: int = 4,
    pred_depth: int = 2,
    mask_ratio: float = 0.5,
    **kwargs: Any,
) -> nn.Module:
    """Self-supervised Audio-JEPA masked predictive transformer encoder."""
    from .jepa import AudioJEPA

    return AudioJEPA(
        embed_dim=embed_dim,
        patch_size=patch_size,
        stride=stride,
        depth=depth,
        pred_depth=pred_depth,
        mask_ratio=mask_ratio,
        **kwargs,
    )



__all__ = [
    "get_model",
    "list_models",
    "register_model",
]
