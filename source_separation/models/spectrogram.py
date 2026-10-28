"""Spectrogram-Domain Single Source Separation & Style Prior Architectures.

Contains 2D convolutional neural network architectures for learning single source separation
and extracting compact acoustic style embeddings directly on 2D time-frequency spectrograms:
  1. ResidualBlock2D, DownsampleBlock2D, UpsampleBlock2D - 2D spatial ConvNet blocks
  2. SpectrogramEncoder, SpectrogramDecoder - Reusable 2D spectrogram encoder-decoder
  3. SingleSourceAutoencoder - Deterministic bottleneck autoencoder with spectral masking
  4. SingleSourceVanillaVAE - Probabilistic Variational Autoencoder (Gaussian posterior + ELBO loss)
  5. SingleSourceBetaVAE - Disentangled β-VAE with Burgess capacity-controlled bottleneck
  6. SpectrogramReconstructionLoss, compute_kl_divergence, compute_vae_loss, train_single_epoch
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. 2D Convolutional Building Blocks
# ---------------------------------------------------------------------------


class ResidualBlock2D(nn.Module):
    """Residual 2D convolutional block with Batch Normalization and LeakyReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(x))


class DownsampleBlock2D(nn.Module):
    """Downsamples spatial frequency-time dimensions by 2x using strided convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock2D(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpsampleBlock2D(nn.Module):
    """Upsamples spatial frequency-time dimensions by 2x using transposed convolution."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ResidualBlock2D(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# 2. Shared Time-Frequency Encoder & Decoder
# ---------------------------------------------------------------------------


class SpectrogramEncoder(nn.Module):
    """Encodes a 2D log-mel mixture spectrogram [B, 1, 64, 128] into spatial features."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.down1 = DownsampleBlock2D(in_channels, base_channels)        # 64x128 -> 32x64
        self.down2 = DownsampleBlock2D(base_channels, base_channels * 2)   # 32x64 -> 16x32
        self.down3 = DownsampleBlock2D(base_channels * 2, base_channels * 4) # 16x32 -> 8x16
        self.down4 = DownsampleBlock2D(base_channels * 4, base_channels * 8) # 8x16 -> 4x8
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.out_dim = (base_channels * 8) * 4 * 8

    def forward(self, x: Tensor) -> Tensor:
        h1 = self.down1(x)
        h2 = self.down2(h1)
        h3 = self.down3(h2)
        h4 = self.down4(h3)
        pooled = self.pool(h4)
        return torch.flatten(pooled, start_dim=1)


class SpectrogramDecoder(nn.Module):
    """Reconstructs target instrument spectrogram [B, 1, 64, 128] from latent embedding z."""

    def __init__(self, out_channels: int = 1, base_channels: int = 32, latent_dim: int = 128) -> None:
        super().__init__()
        self.base_ch = base_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, (base_channels * 8) * 4 * 8),
            nn.ReLU(inplace=True),
        )
        self.up4 = UpsampleBlock2D(base_channels * 8, base_channels * 4) # 4x8 -> 8x16
        self.up3 = UpsampleBlock2D(base_channels * 4, base_channels * 2) # 8x16 -> 16x32
        self.up2 = UpsampleBlock2D(base_channels * 2, base_channels)     # 16x32 -> 32x64
        self.up1 = UpsampleBlock2D(base_channels, base_channels)         # 32x64 -> 64x128
        self.out_head = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.fc(z).view(-1, self.base_ch * 8, 4, 8)
        u4 = self.up4(h)
        u3 = self.up3(u4)
        u2 = self.up2(u3)
        u1 = self.up1(u2)
        return self.out_head(u1)


# ---------------------------------------------------------------------------
# 3. Model 1: Deterministic Single Source Autoencoder
# ---------------------------------------------------------------------------


class SingleSourceAutoencoder(nn.Module):
    """Deterministic Autoencoder for Single Source Separation.

    Compacts the mixture audio spectrogram into a deterministic bottleneck
    representation z in R^D, then reconstructs the isolated single instrument.

    Supports:
      - Direct reconstruction: hat{s} = Decoder(z)
      - Spectral masking: hat{s} = sigmoid(Decoder(z)) * mix_spec
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_mask = use_mask

        self.encoder = SpectrogramEncoder(in_channels=in_channels, base_channels=base_channels)
        self.fc_latent = nn.Linear(self.encoder.out_dim, latent_dim)
        self.decoder = SpectrogramDecoder(
            out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim
        )

    def encode(self, x: Tensor) -> Tensor:
        """Extracts deterministic style embedding z in R^{latent_dim}."""
        feat = self.encoder(x)
        return self.fc_latent(feat)

    def decode(self, z: Tensor, mix: Tensor | None = None) -> Tensor:
        """Decodes latent embedding into separated single instrument spectrogram."""
        raw_out = self.decoder(z)
        if mix is not None and raw_out.shape[-2:] != mix.shape[-2:]:
            raw_out = F.interpolate(
                raw_out, size=mix.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.use_mask and mix is not None:
            mask = torch.sigmoid(raw_out)
            return mask * mix
        return raw_out

    def forward(self, mix: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass taking mixture spectrogram.

        Returns:
            Tuple[Tensor, Tensor]: (reconstructed_instrument, style_embedding_z)
        """
        z = self.encode(mix)
        pred = self.decode(z, mix=mix if self.use_mask else None)
        return pred, z


# ---------------------------------------------------------------------------
# 4. Model 2: Single Source Vanilla VAE
# ---------------------------------------------------------------------------


class SingleSourceVanillaVAE(nn.Module):
    """Probabilistic Variational Autoencoder (VAE) for Single Source Separation.

    Models the latent distribution p(z) ~ N(0, I) and q(z|x) ~ N(mu, diag(sigma^2)).
    Uses the reparameterization trick z = mu + sigma * eps.

    Trained with standard Evidence Lower Bound (ELBO):
      L = L_recon + beta_kl * D_KL(q(z|x) || N(0, I))
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_mask = use_mask

        self.encoder = SpectrogramEncoder(in_channels=in_channels, base_channels=base_channels)
        self.fc_mu = nn.Linear(self.encoder.out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.out_dim, latent_dim)

        self.decoder = SpectrogramDecoder(
            out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes mixture into Gaussian parameters (mu, logvar)."""
        feat = self.encoder(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + sigma * eps."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, mix: Tensor | None = None) -> Tensor:
        raw_out = self.decoder(z)
        if mix is not None and raw_out.shape[-2:] != mix.shape[-2:]:
            raw_out = F.interpolate(
                raw_out, size=mix.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.use_mask and mix is not None:
            mask = torch.sigmoid(raw_out)
            return mask * mix
        return raw_out

    def sample_prior(self, num_samples: int, device: torch.device) -> Tensor:
        """Generates synthetic instrument representations by sampling z ~ N(0, I)."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decoder(z)

    def forward(self, mix: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]:
                (pred_target, mu, logvar, z_sampled)
        """
        mu, logvar = self.encode(mix)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(z, mix=mix if self.use_mask else None)
        return pred, mu, logvar, z


# ---------------------------------------------------------------------------
# 5. Model 3: Advanced Disentangled β-VAE
# ---------------------------------------------------------------------------


class SingleSourceBetaVAE(nn.Module):
    """Advanced Disentangled β-VAE for Single Source Separation & Style Prior Learning.

    Enforces higher regularization on the KL divergence term (beta > 1.0)
    to discover orthogonal, interpretable latent acoustic factors
    (e.g., transient attack rate, harmonic pitch center, spectral timbre).

    Supports capacity-controlled training (Burgess et al., 2018):
      L = L_recon + gamma * | D_KL - C |
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 32,
        beta: float = 4.0,
        capacity: float = 0.0,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.capacity = capacity
        self.use_mask = use_mask

        self.encoder = SpectrogramEncoder(in_channels=in_channels, base_channels=base_channels)
        self.fc_mu = nn.Linear(self.encoder.out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.out_dim, latent_dim)

        self.decoder = SpectrogramDecoder(
            out_channels=in_channels, base_channels=base_channels, latent_dim=latent_dim
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.encoder(x)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, mix: Tensor | None = None) -> Tensor:
        raw_out = self.decoder(z)
        if mix is not None and raw_out.shape[-2:] != mix.shape[-2:]:
            raw_out = F.interpolate(
                raw_out, size=mix.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.use_mask and mix is not None:
            mask = torch.sigmoid(raw_out)
            return mask * mix
        return raw_out

    def latent_traversal(
        self,
        base_z: Tensor,
        dim_idx: int,
        values: Tensor,
        mix: Tensor | None = None,
    ) -> Tensor:
        """Sweeps a single latent dimension while holding all other dimensions constant.

        Args:
            base_z: [1, latent_dim] base reference latent vector.
            dim_idx: Index of dimension to traverse (0 <= dim_idx < latent_dim).
            values: 1D tensor of sweep values (e.g. linspace(-3.0, 3.0, 7)).
            mix: Optional mixture spectrogram for spectral masking.
        Returns:
            [N_values, 1, 64, 128] traversed reconstructed spectrograms.
        """
        N = len(values)
        traversed_z = base_z.repeat(N, 1)
        traversed_z[:, dim_idx] = values
        mix_rep = mix.repeat(N, 1, 1, 1) if mix is not None else None
        return self.decode(traversed_z, mix=mix_rep)

    def forward(self, mix: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(mix)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(z, mix=mix if self.use_mask else None)
        return pred, mu, logvar, z


# ---------------------------------------------------------------------------
# 6. Loss Functions & Training Utilities
# ---------------------------------------------------------------------------


class SpectrogramReconstructionLoss(nn.Module):
    """Combined L1 Loss and Multi-Scale Spectral Frobenius Convergence Loss."""

    def __init__(self, alpha_sc: float = 0.5) -> None:
        super().__init__()
        self.alpha_sc = alpha_sc

    def forward(self, pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        l1 = F.l1_loss(pred, target)
        # Spectral Convergence Frobenius norm
        diff_norm = torch.norm(target - pred, p="fro", dim=(-2, -1))
        target_norm = torch.norm(target, p="fro", dim=(-2, -1)) + 1e-6
        sc = torch.mean(diff_norm / target_norm)
        total = l1 + self.alpha_sc * sc
        return total, l1, sc


def compute_kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Computes mean analytical KL divergence D_KL(q(z|x) || N(0, I))."""
    kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return torch.mean(kl_per_sample)


def compute_vae_loss(
    pred: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    recon_loss_fn: SpectrogramReconstructionLoss,
    beta: float = 1.0,
    capacity: float = 0.0,
) -> dict[str, Tensor]:
    """Computes full VAE / β-VAE objective."""
    total_recon, l1, sc = recon_loss_fn(pred, target)
    kl = compute_kl_divergence(mu, logvar)

    if capacity > 0:
        kl_term = beta * torch.abs(kl - capacity)
    else:
        kl_term = beta * kl

    total_loss = total_recon + kl_term
    return {
        "loss": total_loss,
        "recon_loss": total_recon,
        "l1_loss": l1,
        "sc_loss": sc,
        "kl_loss": kl,
        "weighted_kl": kl_term,
    }


def train_single_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "ae",  # "ae", "vae", "beta_vae"
    beta: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Trains model for one epoch and returns averaged metrics."""
    model.train()
    recon_loss_fn = SpectrogramReconstructionLoss()
    total_loss_accum = 0.0
    recon_loss_accum = 0.0
    kl_loss_accum = 0.0
    batches = 0

    for batch in dataloader:
        if max_batches is not None and batches >= max_batches:
            break
        mix = batch["mix_spec"].to(device)
        target = batch["target_spec"].to(device)
        optimizer.zero_grad()

        if model_type == "ae":
            pred, z = model(mix)
            total_loss, l1, sc = recon_loss_fn(pred, target)
            kl = torch.tensor(0.0)
        else:
            pred, mu, logvar, z = model(mix)
            loss_dict = compute_vae_loss(
                pred=pred,
                target=target,
                mu=mu,
                logvar=logvar,
                recon_loss_fn=recon_loss_fn,
                beta=beta if model_type == "beta_vae" else 1.0,
            )
            total_loss = loss_dict["loss"]
            recon_loss = loss_dict["recon_loss"]
            kl = loss_dict["kl_loss"]

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += total_loss.item()
        recon_loss_accum += (recon_loss.item() if model_type != "ae" else total_loss.item())
        kl_loss_accum += kl.item()
        batches += 1

    return {
        "loss": total_loss_accum / max(1, batches),
        "recon": recon_loss_accum / max(1, batches),
        "kl": kl_loss_accum / max(1, batches),
    }


__all__ = [
    "DownsampleBlock2D",
    "ResidualBlock2D",
    "SingleSourceAutoencoder",
    "SingleSourceBetaVAE",
    "SingleSourceVanillaVAE",
    "SpectrogramDecoder",
    "SpectrogramEncoder",
    "SpectrogramReconstructionLoss",
    "UpsampleBlock2D",
    "compute_kl_divergence",
    "compute_vae_loss",
    "train_single_epoch",
]
