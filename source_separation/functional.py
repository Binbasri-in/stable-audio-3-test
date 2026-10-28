"""Stateless functional operations for audio source separation.

Provides pure functional operations following the PyTorch ``torch.nn.functional``
paradigm: metrics, spectral distance formulas, divergence measures, and signal transformations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def snr(
    estimate: Tensor,
    reference: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Computes Signal-to-Noise Ratio (SNR) in decibels (dB).

    Formula:
        SNR = 10 * log10( ||reference||^2 / (||reference - estimate||^2 + eps) )

    Args:
        estimate: Model estimated audio [..., T].
        reference: Reference clean audio [..., T].
        eps: Numerical stability constant.

    Returns:
        Tensor of SNR values in dB, reduced over the temporal dimension.
    """
    noise = reference - estimate
    signal_power = torch.sum(reference**2, dim=-1)
    noise_power = torch.sum(noise**2, dim=-1)
    res = 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))
    if res.ndim > 1:
        res = res.mean(dim=-1)
    return res


def si_sdr(
    estimate: Tensor,
    reference: Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tensor:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).

    Args:
        estimate: Model separated estimation [..., T].
        reference: Reference ground truth clean source [..., T].
        eps: Numerical stability constant.
        zero_mean: Whether to subtract sample mean along time.

    Returns:
        Tensor of SI-SDR values in dB per sample in batch.
    """
    if zero_mean:
        reference = reference - torch.mean(reference, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference**2, dim=-1, keepdim=True) + eps
    alpha = dot_product / ref_energy

    s_target = alpha * reference
    e_noise = estimate - s_target

    target_power = torch.sum(s_target**2, dim=-1)
    noise_power = torch.sum(e_noise**2, dim=-1)

    res = 10.0 * torch.log10((target_power + eps) / (noise_power + eps))
    if res.ndim > 1:
        res = res.mean(dim=-1)
    return res


def si_sdr_improvement(
    estimate: Tensor,
    reference: Tensor,
    mixture: Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tensor:
    """Computes SI-SDR Improvement (SI-SDRi) relative to the unprocessed mixture."""
    return si_sdr(estimate, reference, eps=eps, zero_mean=zero_mean) - si_sdr(
        mixture, reference, eps=eps, zero_mean=zero_mean
    )


def spectral_convergence(
    pred_mag: Tensor,
    target_mag: Tensor,
    eps: float = 1e-2,
) -> Tensor:
    """Computes Frobenius-norm spectral convergence between predicted and target magnitudes."""
    diff = torch.norm(target_mag - pred_mag, p="fro", dim=(-2, -1))
    norm = torch.clamp(torch.norm(target_mag, p="fro", dim=(-2, -1)), min=eps)
    sc = torch.clamp(diff / norm, max=20.0)
    return torch.mean(sc)


def log_stft_magnitude(
    pred_mag: Tensor,
    target_mag: Tensor,
    eps: float = 1e-4,
) -> Tensor:
    """Computes L1 distance between log-magnitude spectrograms."""
    log_pred = torch.log(torch.clamp(pred_mag, min=eps))
    log_target = torch.log(torch.clamp(target_mag, min=eps))
    return F.l1_loss(log_pred, log_target)


def kl_divergence_normal(
    mu: Tensor,
    logvar: Tensor,
) -> Tensor:
    """Computes analytic Kullback-Leibler divergence between N(mu, diag(exp(logvar))) and N(0, I).

    Formula:
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl / mu.size(0)


def mixture_consistency(
    estimates: Tensor,
    mixture: Tensor,
) -> Tensor:
    """Applies projection onto the mixture-consistent linear subspace (Wisdom et al., 2021).

    Ensures that sum_j (estimates_j) == mixture by distributing the residual difference
    equally across all sources.

    Args:
        estimates: Source estimates of shape [B, J, C, T] or [J, C, T].
        mixture: Mixture audio of shape [B, C, T] or [C, T].

    Returns:
        Tensor of consistent source estimates with identical shape to estimates.
    """
    if estimates.ndim == 4:
        # [B, J, C, T]
        num_sources = estimates.shape[1]
        mix_expanded = mixture.unsqueeze(1)
        est_sum = estimates.sum(dim=1, keepdim=True)
        residual = (mix_expanded - est_sum) / num_sources
        return estimates + residual
    elif estimates.ndim == 3:
        # [J, C, T]
        num_sources = estimates.shape[0]
        mix_expanded = mixture.unsqueeze(0)
        est_sum = estimates.sum(dim=0, keepdim=True)
        residual = (mix_expanded - est_sum) / num_sources
        return estimates + residual
    else:
        raise ValueError(f"Expected estimates tensor with 3 or 4 dimensions, got {estimates.ndim}")


__all__ = [
    "kl_divergence_normal",
    "log_stft_magnitude",
    "mixture_consistency",
    "si_sdr",
    "si_sdr_improvement",
    "snr",
    "spectral_convergence",
]
