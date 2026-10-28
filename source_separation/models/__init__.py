"""Neural Network Architectures for Audio Source Separation and Acoustic Prior Extraction.

Provides modular, high-performance deep learning model architectures operating on raw 1D waveforms
and 2D time-frequency spectrograms:

Submodules:
  1. `autoencoder` (`autoencoders`):
     - `WaveformAutoencoder`: 1D strided convolutional autoencoder with physical pressure preservation.
     - `WaveformUNetAutoencoder`: 1D waveform autoencoder with U-Net multi-scale skip connections.
     - `SingleSourceAutoencoder`: 2D spectrogram bottleneck autoencoder with spectral masking.
  2. `vae`:
     - `WaveformVanillaVAE`: Time-domain variational autoencoder with analytical Gaussian posterior.
     - `WaveformBetaVAE`: Disentangled β-VAE with capacity control for acoustic factor factorization.
     - `SingleSourceVanillaVAE`, `SingleSourceBetaVAE`: 2D spectrogram variational autoencoders.
  3. `spectrogram`:
     - 2D convolutional blocks and spectrogram-domain autoencoders and VAEs.
  4. `glow`:
     - `WaveformGlow`: Invertible normalizing flow with 1x1 LU-decomposed convolutions and WaveNet coupling.
     - `GlowMAPSeparator`: Maximum A Posteriori single-source inversion solver.
  5. `ebm`:
     - `ScalarLatentEBM`: Deep scalar energy model over compressed latent representations.
     - `EBMReplayBuffer`: Persistent MCMC sample replay buffer.
     - `CompositionalLangevinSeparator`: Test-time product-of-experts energy composition demixer.
  6. `jem`:
     - `WaveformJEM`: Unified discriminative classifier and generative free-energy model.
     - `JEMReplayBuffer`: Replay buffer for short-run SGLD negative sampling.
     - `JEMSingleSourceSeparator`: Iterative single-source separation with bleed suppression barrier.
     - `JEMGenerativeSteeringBridge`: Projects JEM acoustic scores into diffusion generative guidance.
  7. `jepa`:
     - `AudioJEPA`: Joint-Embedding Predictive Architecture trained via masked patch feature prediction.
     - `JEPAGuidedSeparationHead`: FiLM-modulated waveform decoder conditioned on invariant JEPA priors.
"""

from __future__ import annotations

from . import autoencoder, autoencoders, ebm, glow, jem, jepa, registry, spectrogram, vae
from .autoencoder import (
    DownsampleBlock1D,
    ResidualBlock1D,
    UpsampleBlock1D,
    WaveformAutoencoder,
    WaveformUNetAutoencoder,
)
from .ebm import (
    CompositionalLangevinSeparator,
    EBMReplayBuffer,
    ScalarLatentEBM,
    SmoothResidualBlock1D,
    sample_langevin_mcmc,
)
from .glow import (
    ActNorm1D,
    DilatedResidualLayer1D,
    GlowFlowStep1D,
    GlowMAPSeparator,
    Invertible1x1Conv1D,
    Squeeze1D,
    WaveformGlow,
    WaveNetCouplingBlock1D,
)
from .jem import (
    JEMGenerativeSteeringBridge,
    JEMReplayBuffer,
    JEMResidualBlock1D,
    JEMSingleSourceSeparator,
    WaveformJEM,
)
from .jepa import (
    AudioJEPA,
    AudioPatchEmbed,
    JEPAEncoderBlock,
    JEPAGuidedSeparationHead,
)
from .registry import (
    get_model,
    list_models,
    register_model,
)
from .spectrogram import (
    DownsampleBlock2D,
    ResidualBlock2D,
    SingleSourceAutoencoder,
    SingleSourceBetaVAE,
    SingleSourceVanillaVAE,
    SpectrogramDecoder,
    SpectrogramEncoder,
    SpectrogramReconstructionLoss,
    UpsampleBlock2D,
    compute_kl_divergence,
    compute_vae_loss,
    train_single_epoch,
)
from .vae import (
    WaveformBetaVAE,
    WaveformVanillaVAE,
)

__all__ = [
    # Submodules
    "autoencoder",
    "autoencoders",
    "vae",
    "spectrogram",
    "glow",
    "ebm",
    "jem",
    "jepa",
    "registry",
    # Model Factory & Registry
    "get_model",
    "list_models",
    "register_model",
    # 1D Waveform Autoencoders
    "ResidualBlock1D",
    "DownsampleBlock1D",
    "UpsampleBlock1D",
    "WaveformAutoencoder",
    "WaveformUNetAutoencoder",
    # 1D Waveform VAEs
    "WaveformVanillaVAE",
    "WaveformBetaVAE",
    # 2D Spectrogram Models
    "ResidualBlock2D",
    "DownsampleBlock2D",
    "UpsampleBlock2D",
    "SpectrogramEncoder",
    "SpectrogramDecoder",
    "SingleSourceAutoencoder",
    "SingleSourceVanillaVAE",
    "SingleSourceBetaVAE",
    "SpectrogramReconstructionLoss",
    "compute_kl_divergence",
    "compute_vae_loss",
    "train_single_epoch",
    # Waveform Glow Normalizing Flows
    "Squeeze1D",
    "ActNorm1D",
    "Invertible1x1Conv1D",
    "DilatedResidualLayer1D",
    "WaveNetCouplingBlock1D",
    "GlowFlowStep1D",
    "WaveformGlow",
    "GlowMAPSeparator",
    # Energy-Based Models
    "SmoothResidualBlock1D",
    "ScalarLatentEBM",
    "EBMReplayBuffer",
    "sample_langevin_mcmc",
    "CompositionalLangevinSeparator",
    # Joint Energy Models
    "JEMResidualBlock1D",
    "WaveformJEM",
    "JEMReplayBuffer",
    "JEMSingleSourceSeparator",
    "JEMGenerativeSteeringBridge",
    # Audio-JEPA
    "AudioPatchEmbed",
    "JEPAEncoderBlock",
    "AudioJEPA",
    "JEPAGuidedSeparationHead",
]
