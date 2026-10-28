"""Single Source Separation Waveform & Spectrogram Variational Autoencoder (VAE) Architectures.

Provides probabilistic latent variable models for waveform-domain and spectrogram-domain
single-source separation, acoustic prior density learning, and generative sampling:
  1. WaveformVanillaVAE: 1D temporal convolutional encoder projecting to Gaussian
     posterior parameters (mu, logvar), reparameterization trick, and symmetric
     transposed convolutional decoder back to raw waveform.
  2. WaveformBetaVAE: Disentangled β-VAE with capacity control for learning orthogonal
     acoustic prior factors and suppressing cross-stem harmonic bleed.
  3. SingleSourceVanillaVAE, SingleSourceBetaVAE: 2D spectrogram VAE architectures.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .autoencoder import DownsampleBlock1D, UpsampleBlock1D
from .spectrogram import (
    SingleSourceBetaVAE,
    SingleSourceVanillaVAE,
    compute_kl_divergence,
    compute_vae_loss,
)


class WaveformVanillaVAE(nn.Module):
    """Time-Domain Waveform Variational Autoencoder for Single Source Separation.

    Encodes mixture audio [B, in_channels, T] into approximate posterior
    distribution parameters q_phi(z | x) = N(mu, diag(sigma^2)), samples z
    via reparameterization, and decodes back to the isolated target source
    waveform [B, in_channels, T].

    Operates directly in the time domain to maintain exact linear physical
    superposition and eliminate phase distortion artifacts.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        strides: tuple[int, ...] = (4, 4, 4, 4),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.strides = strides

        # Total downsampling factor: 4^4 = 256x
        self.enc1 = DownsampleBlock1D(in_channels, base_channels, stride=strides[0])
        self.enc2 = DownsampleBlock1D(base_channels, base_channels * 2, stride=strides[1])
        self.enc3 = DownsampleBlock1D(base_channels * 2, base_channels * 4, stride=strides[2])
        self.enc4 = DownsampleBlock1D(base_channels * 4, base_channels * 8, stride=strides[3])

        # Latent Gaussian distribution heads: mean (mu) and log-variance (logvar)
        enc_out_channels = base_channels * 8
        self.fc_mu = nn.Sequential(
            nn.Conv1d(enc_out_channels, latent_dim, kernel_size=3, padding=1),
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv1d(enc_out_channels, latent_dim, kernel_size=3, padding=1),
        )

        # Decoder: Latent projection and 4 symmetric transposed upsampling stages
        self.dec_proj = nn.Sequential(
            nn.Conv1d(latent_dim, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec4 = UpsampleBlock1D(base_channels * 8, base_channels * 4, stride=strides[3])
        self.dec3 = UpsampleBlock1D(base_channels * 4, base_channels * 2, stride=strides[2])
        self.dec2 = UpsampleBlock1D(base_channels * 2, base_channels, stride=strides[1])
        self.dec1 = nn.ConvTranspose1d(
            base_channels,
            in_channels,
            kernel_size=8,
            stride=strides[0],
            padding=4,
            bias=True,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes mixture waveform into Gaussian latent parameters (mu, logvar).

        Args:
            x: [B, C, T] audio waveform tensor.

        Returns:
            mu: [B, latent_dim, T_latent] posterior mean.
            logvar: [B, latent_dim, T_latent] posterior log-variance.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        mu = self.fc_mu(e4)
        logvar = self.fc_logvar(e4)
        # Numerical clamp on log-variance to avoid gradient explosion
        logvar = torch.clamp(logvar, min=-15.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Applies the Reparameterization Trick: z = mu + eps * sigma."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, target_len: int | None = None) -> Tensor:
        """Decodes latent features z into reconstructed source waveform."""
        d4 = self.dec_proj(z)
        d3 = self.dec4(d4)
        d2 = self.dec3(d3)
        d1 = self.dec2(d2)
        out = self.dec1(d1)

        # Precise sample length alignment
        if target_len is not None:
            curr_len = out.shape[-1]
            if curr_len > target_len:
                out = out[..., :target_len]
            elif curr_len < target_len:
                out = F.pad(out, (0, target_len - curr_len))

        return torch.tanh(out)

    def forward(
        self,
        mix: Tensor,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass taking input mixture waveform.

        Args:
            mix: [B, C, T] mixture audio tensor.
            deterministic: If True, bypasses sampling and sets z = mu.

        Returns:
            Dict containing:
                'recon': [B, C, T] reconstructed target stem waveform.
                'mu': [B, latent_dim, T_latent] posterior mean.
                'logvar': [B, latent_dim, T_latent] posterior log-variance.
                'latent': [B, latent_dim, T_latent] sampled latent vector z.
                'residual': [B, C, T] accompaniment residual: mix - recon.
        """
        target_len = mix.shape[-1]
        mu, logvar = self.encode(mix)

        if deterministic or not self.training:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)

        recon = self.decode(z, target_len=target_len)
        residual = mix - recon

        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "residual": residual,
        }

    @torch.no_grad()
    def sample_prior(
        self,
        num_samples: int = 1,
        target_len: int = 66150,
        device: torch.device | None = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Unconditionally samples synthetic instrument waveforms from prior p(z) = N(0, I)."""
        dev = device or next(self.parameters()).device
        total_downsample = math.prod(self.strides)
        latent_frames = math.ceil(target_len / total_downsample)

        z = torch.randn(num_samples, self.latent_dim, latent_frames, device=dev) * temperature
        return self.decode(z, target_len=target_len)

    @torch.no_grad()
    def interpolate(
        self,
        mix1: Tensor,
        mix2: Tensor,
        num_steps: int = 5,
        target_len: int | None = None,
    ) -> Tensor:
        """Performs linear latent space interpolation between two audio mixtures."""
        mu1, _ = self.encode(mix1)
        mu2, _ = self.encode(mix2)

        min_len = min(mu1.shape[-1], mu2.shape[-1])
        mu1 = mu1[..., :min_len]
        mu2 = mu2[..., :min_len]

        t_len = target_len or mix1.shape[-1]
        alphas = torch.linspace(0.0, 1.0, num_steps, device=mix1.device)
        interpolated_stems = []

        for alpha in alphas:
            z_interp = (1.0 - alpha) * mu1 + alpha * mu2
            recon_step = self.decode(z_interp, target_len=t_len)
            interpolated_stems.append(recon_step)

        return torch.cat(interpolated_stems, dim=0)


class WaveformBetaVAE(WaveformVanillaVAE):
    """Disentangled Beta-VAE for Waveform Single Source Separation."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
        strides: tuple[int, ...] = (4, 4, 4, 4),
        beta: float = 4.0,
        capacity: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_dim=latent_dim,
            strides=strides,
        )
        self.beta = beta
        self.capacity = capacity

    @torch.no_grad()
    def latent_traversal(
        self,
        mix: Tensor,
        dim_idx: int,
        values: Tensor,
        target_len: int | None = None,
    ) -> Tensor:
        """Sweeps a single latent dimension while holding all other latent dimensions frozen."""
        mu, _ = self.encode(mix)
        N = len(values)
        t_len = target_len or mix.shape[-1]

        traversed_mu = mu.repeat(N, 1, 1)
        for i, val in enumerate(values):
            traversed_mu[i, dim_idx, :] = val

        return self.decode(traversed_mu, target_len=t_len)


__all__ = [
    # 1D Waveform VAEs
    "WaveformVanillaVAE",
    "WaveformBetaVAE",
    # 2D Spectrogram VAEs
    "SingleSourceVanillaVAE",
    "SingleSourceBetaVAE",
    "compute_kl_divergence",
    "compute_vae_loss",
]
