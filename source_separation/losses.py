"""Stateful loss functions and optimization objectives for audio source separation.

Provides composite, multi-resolution spectral, generative, and energy-based objectives
following standard PyTorch ``torch.nn.modules.loss`` conventions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .functional import (
    kl_divergence_normal,
    log_stft_magnitude,
    spectral_convergence,
)


class SpectralConvergenceLoss(nn.Module):
    """Frobenius-norm spectral convergence between predicted and target magnitudes."""

    def __init__(self, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        return spectral_convergence(pred_mag, target_mag, eps=self.eps)


class LogSTFTMagnitudeLoss(nn.Module):
    """L1 distance between log-magnitude spectrograms."""

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        return log_stft_magnitude(pred_mag, target_mag, eps=self.eps)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss for raw waveform source separation.

    Evaluates time-frequency spectral alignment across multiple window lengths
    (e.g., 512, 1024, 2048) to capture both sharp transient attacks and fine harmonic pitch.
    """

    def __init__(
        self,
        fft_sizes: tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: tuple[int, ...] = (128, 256, 512),
        win_lengths: tuple[int, ...] = (512, 1024, 2048),
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sc_loss = SpectralConvergenceLoss()
        self.log_loss = LogSTFTMagnitudeLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Computes multi-resolution STFT loss between predicted and target waveforms.

        Args:
            pred: Predicted audio waveform [B, C, T] or [B, T].
            target: Ground truth audio waveform [B, C, T] or [B, T].
        """
        orig_device = pred.device
        if orig_device.type == "mps":
            pred = pred.to("cpu")
            target = target.to("cpu")

        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        compute_device = pred.device
        total_loss = torch.tensor(0.0, device=compute_device)
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, strict=False):
            window = torch.hann_window(win, device=compute_device)
            stft_pred = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                return_complex=True,
            )
            stft_target = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                return_complex=True,
            )
            mag_pred = torch.abs(stft_pred)
            mag_target = torch.abs(stft_target)

            sc = self.sc_loss(mag_pred, mag_target)
            log_l = self.log_loss(mag_pred, mag_target)
            total_loss = total_loss + sc + log_l

        res = total_loss / len(self.fft_sizes)
        return res.to(orig_device)


class SingleSourceSeparationLoss(nn.Module):
    """Composite loss balancing time-domain L1 error and Multi-Resolution Spectral Loss."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(self, pred: Tensor | dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        if isinstance(pred, dict):
            pred = pred["recon"]
        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        total = self.alpha_l1 * l1 + self.alpha_stft * stft
        return {
            "total_loss": total,
            "l1_loss": l1,
            "stft_loss": stft,
        }


class SingleSourceVAELoss(nn.Module):
    """Composite loss for Single Source Separation VAEs: L1 + Multi-Resolution STFT + KL Divergence."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
        beta_kl: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.beta_kl = beta_kl
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_or_out: Tensor | dict[str, Tensor],
        target: Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(pred_or_out, dict):
            pred = pred_or_out["recon"]
            mu = pred_or_out.get("mu", None)
            logvar = pred_or_out.get("logvar", None)
        else:
            pred = pred_or_out
            mu = None
            logvar = None

        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        recon_loss = self.alpha_l1 * l1 + self.alpha_stft * stft

        if mu is not None and logvar is not None:
            kl = kl_divergence_normal(mu, logvar)
        else:
            kl = torch.tensor(0.0, device=pred.device)

        total = recon_loss + self.beta_kl * kl

        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "l1_loss": l1,
            "stft_loss": stft,
            "kl_loss": kl,
            "weighted_kl": self.beta_kl * kl,
        }


class SingleSourceBetaVAELoss(nn.Module):
    """Composite loss for Single Source Separation Beta-VAE (Higgins et al., 2017; Burgess et al., 2018).

    Formulation:
        L = alpha_l1 * L1 + alpha_stft * MR-STFT + beta * |D_KL - C|
    """

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
        beta: float = 4.0,
        beta_scale: float = 1e-4,
        capacity: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.beta = beta
        self.beta_scale = beta_scale
        self.capacity = capacity
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_or_out: Tensor | dict[str, Tensor],
        target: Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(pred_or_out, dict):
            pred = pred_or_out["recon"]
            mu = pred_or_out.get("mu", None)
            logvar = pred_or_out.get("logvar", None)
        else:
            pred = pred_or_out
            mu = None
            logvar = None

        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        recon_loss = self.alpha_l1 * l1 + self.alpha_stft * stft

        if mu is not None and logvar is not None:
            kl = kl_divergence_normal(mu, logvar)
        else:
            kl = torch.tensor(0.0, device=pred.device)

        eff_beta = self.beta * self.beta_scale
        if self.capacity > 0:
            kl_term = eff_beta * torch.abs(kl - self.capacity)
        else:
            kl_term = eff_beta * kl

        total = recon_loss + kl_term

        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "l1_loss": l1,
            "stft_loss": stft,
            "kl_loss": kl,
            "weighted_kl": kl_term,
            "eff_beta": torch.tensor(eff_beta, device=pred.device),
            "capacity": torch.tensor(self.capacity, device=pred.device),
        }


class LatentContrastiveDivergenceLoss(nn.Module):
    """Contrastive Divergence (CD) loss with L2 energy regularization for ScalarLatentEBM."""

    def __init__(self, energy_reg: float = 0.05) -> None:
        super().__init__()
        self.energy_reg = energy_reg

    def forward(self, pos_energy: Tensor, neg_energy: Tensor) -> dict[str, Tensor]:
        pos_mean = pos_energy.mean()
        neg_mean = neg_energy.mean()
        cd_loss = pos_mean - neg_mean
        reg_loss = self.energy_reg * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())
        total = cd_loss + reg_loss
        return {
            "total_loss": total,
            "cd_loss": cd_loss,
            "reg_loss": reg_loss,
            "pos_energy": pos_mean,
            "neg_energy": neg_mean,
            "energy_gap": neg_mean - pos_mean,
        }


class WaveformJEMLoss(nn.Module):
    """Joint Energy Model (JEM) objective balancing classification and free-energy CD."""

    def __init__(self, alpha_energy: float = 0.2, energy_reg: float = 0.05) -> None:
        super().__init__()
        self.alpha_energy = alpha_energy
        self.energy_reg = energy_reg
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        pos_energy: Tensor,
        neg_energy: Tensor,
    ) -> dict[str, Tensor]:
        clf_loss = self.ce_loss(logits, targets)
        pos_mean = pos_energy.mean()
        neg_mean = neg_energy.mean()
        cd_loss = pos_mean - neg_mean
        reg_loss = self.energy_reg * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())
        energy_loss = cd_loss + reg_loss
        total = clf_loss + self.alpha_energy * energy_loss

        return {
            "total_loss": total,
            "clf_loss": clf_loss,
            "energy_loss": energy_loss,
            "cd_loss": cd_loss,
            "reg_loss": reg_loss,
            "pos_energy": pos_mean,
            "neg_energy": neg_mean,
            "energy_gap": neg_mean - pos_mean,
        }


__all__ = [
    "LatentContrastiveDivergenceLoss",
    "LogSTFTMagnitudeLoss",
    "MultiResolutionSTFTLoss",
    "SingleSourceBetaVAELoss",
    "SingleSourceSeparationLoss",
    "SingleSourceVAELoss",
    "SpectralConvergenceLoss",
    "WaveformJEMLoss",
]
