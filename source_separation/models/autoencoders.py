"""Autoencoders alias module."""

from __future__ import annotations

from .autoencoder import (
    DownsampleBlock1D,
    DownsampleBlock2D,
    ResidualBlock1D,
    ResidualBlock2D,
    SingleSourceAutoencoder,
    SpectrogramDecoder,
    SpectrogramEncoder,
    UpsampleBlock1D,
    UpsampleBlock2D,
    WaveformAutoencoder,
    WaveformUNetAutoencoder,
)

__all__ = [
    "DownsampleBlock1D",
    "DownsampleBlock2D",
    "ResidualBlock1D",
    "ResidualBlock2D",
    "SingleSourceAutoencoder",
    "SpectrogramDecoder",
    "SpectrogramEncoder",
    "UpsampleBlock1D",
    "UpsampleBlock2D",
    "WaveformAutoencoder",
    "WaveformUNetAutoencoder",
]
