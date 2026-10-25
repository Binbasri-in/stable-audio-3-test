"""Latent EBM Trainer alias."""

from __future__ import annotations

from .ebm import LatentContrastiveDivergenceLoss, LatentEBMTrainer

__all__ = [
    "LatentContrastiveDivergenceLoss",
    "LatentEBMTrainer",
]
