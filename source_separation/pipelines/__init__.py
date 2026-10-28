"""Inference pipelines and iterative optimization solvers for audio source separation.

Provides decoupled inference workflows (Langevin dynamics, MAP estimation, test-time gradient
ascent, cascaded multi-stage separation) following the PyTorch ecosystem pipeline paradigm.
"""

from __future__ import annotations

from ..evaluation.cdx import CascadedCinematicDemixer
from ..models.ebm import CompositionalLangevinSeparator
from ..models.glow import GlowMAPSeparator
from ..models.jem import JEMSingleSourceSeparator
from .base import BaseAudioPipeline

# Backward compatible aliases
GlowWaveformSeparator = GlowMAPSeparator
LatentEBMSeparator = CompositionalLangevinSeparator
JEMGradientSeparator = JEMSingleSourceSeparator

__all__ = [
    "BaseAudioPipeline",
    "CascadedCinematicDemixer",
    "CompositionalLangevinSeparator",
    "GlowMAPSeparator",
    "GlowWaveformSeparator",
    "JEMGradientSeparator",
    "JEMSingleSourceSeparator",
    "LatentEBMSeparator",
]

