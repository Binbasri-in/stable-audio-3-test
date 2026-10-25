"""Audio Source Separation Trainers Package.

Provides dedicated, modular trainer implementations and objective functions for
various audio source separation and self-supervised model architectures:

1. `single_source`: Universal Architecture-Agnostic Single Source Separation Trainer
   - `SingleSourceTrainer`: Works with deterministic autoencoders, VAEs, Beta-VAEs, U-Nets, and Flows.
   - `SingleSourceSeparationLoss`: Composite L1 time-domain waveform error + Multi-Resolution STFT loss.
   - `SingleSourceVAELoss`: L1 + Multi-Resolution STFT + Analytical Gaussian KL divergence.
   - `SingleSourceBetaVAELoss`: L1 + Multi-Resolution STFT + beta-scaled / capacity-controlled KL penalty.

2. `ebm` (`latent_ebm`): Energy-Based Model Training via Contrastive Divergence
   - `LatentEBMTrainer`: Trains scalar energy estimators on latent representations using Langevin MCMC.
   - `LatentContrastiveDivergenceLoss`: Positive vs negative energy contrastive loss with L2 regularization.

3. `jem` (`waveform_jem`): Joint Energy Model Multi-Source Training
   - `WaveformJEMTrainer`: Joint classification and generative free-energy learning with SGLD dynamics.
   - `WaveformJEMLoss`: Joint cross-entropy and free-energy contrastive divergence.

4. `audio_jepa`: Self-Supervised Audio-JEPA Pre-training
   - `AudioJEPATrainer`: Masked latent prediction and EMA target encoder updates.

5. `jepa_separation`: Downstream Supervised JEPA Separation Head Training
   - `JEPASeparationTrainer`: Trains separation head conditioned on frozen JEPA representations.

6. `base`: Common trainer base class, logging utilities, and reconstruction extractors.
"""

from __future__ import annotations

from .audio_jepa import AudioJEPATrainer
from .base import BaseTrainer, extract_reconstruction, logger
from .ebm import LatentContrastiveDivergenceLoss, LatentEBMTrainer
from .jem import WaveformJEMLoss, WaveformJEMTrainer
from .jepa_separation import JEPASeparationTrainer
from .single_source import (
    SingleSourceBetaVAELoss,
    SingleSourceSeparationLoss,
    SingleSourceTrainer,
    SingleSourceVAELoss,
)

__all__ = [
    # Base and common utilities
    "BaseTrainer",
    "extract_reconstruction",
    "logger",
    # Single-source trainers and losses
    "SingleSourceSeparationLoss",
    "SingleSourceVAELoss",
    "SingleSourceBetaVAELoss",
    "SingleSourceTrainer",
    # Energy-Based Model trainers and losses
    "LatentContrastiveDivergenceLoss",
    "LatentEBMTrainer",
    # Joint Energy Model trainers and losses
    "WaveformJEMLoss",
    "WaveformJEMTrainer",
    # Audio-JEPA trainers
    "AudioJEPATrainer",
    "JEPASeparationTrainer",
]
