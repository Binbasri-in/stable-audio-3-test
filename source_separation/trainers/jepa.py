"""JEPA Trainers Module.

Re-exports AudioJEPATrainer (self-supervised masked latent representation learning)
and JEPASeparationTrainer (downstream supervised separation head training).
"""

from __future__ import annotations

from .audio_jepa import AudioJEPATrainer
from .jepa_separation import JEPASeparationTrainer

__all__ = [
    "AudioJEPATrainer",
    "JEPASeparationTrainer",
]
