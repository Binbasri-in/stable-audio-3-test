"""Fundamental source separation metrics, multi-resolution losses, and audio audit tools."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import to_numpy_channels_first, to_tensor_channels_first


# ===========================================================================
# 1. Core Fundamental Separation Metrics
# ===========================================================================


def compute_snr(
    target: Tensor,
    estimate: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Computes standard Signal-to-Noise Ratio (SNR) in decibels (dB).

    Formula:
        SNR = 10 * log10( ||target||^2 / (||target - estimate||^2 + eps) )

    Args:
        target: Reference ground truth audio [B, C, T] or [B, T].
        estimate: Model estimated audio [B, C, T] or [B, T].
        eps: Small numerical stability constant.

    Returns:
        [B] SNR in dB per batch item.
    """
    noise = target - estimate
    signal_power = torch.sum(target ** 2, dim=-1)
    noise_power = torch.sum(noise ** 2, dim=-1)
    snr = 10.0 * torch.log10((signal_power + eps) / (noise_power + eps))
    if snr.ndim > 1:
        snr = snr.mean(dim=-1)
    return snr


def compute_si_sdr(
    target: Tensor,
    estimate: Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tensor:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).

    SI-SDR (Le Roux et al., 2019) is the standard benchmark metric for
    evaluating audio source separation. Unlike classic SNR, SI-SDR is invariant
    to arbitrary global scale factors alpha, isolating true separation fidelity
    from overall volume changes.

    Mathematical formulation:
        target_zm = target - mean(target)
        estimate_zm = estimate - mean(estimate)
        alpha = <estimate_zm, target_zm> / (||target_zm||^2 + eps)
        s_target = alpha * target_zm
        e_noise = estimate_zm - s_target
        SI-SDR = 10 * log10( ||s_target||^2 / (||e_noise||^2 + eps) )

    Args:
        target: Reference ground truth clean source [B, C, T] or [B, T].
        estimate: Model separated estimation [B, C, T] or [B, T].
        eps: Numerical stability constant.
        zero_mean: Whether to subtract sample mean before projection.

    Returns:
        [B] SI-SDR in dB for each sample in the batch.
    """
    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # Compute orthogonal projection scaling factor alpha
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / target_energy

    # Decompose into target signal and residual distortion noise
    s_target = alpha * target
    e_noise = estimate - s_target

    target_power = torch.sum(s_target ** 2, dim=-1)
    noise_power = torch.sum(e_noise ** 2, dim=-1)

    si_sdr = 10.0 * torch.log10((target_power + eps) / (noise_power + eps))
    if si_sdr.ndim > 1:
        si_sdr = si_sdr.mean(dim=-1)
    return si_sdr


def compute_si_sdr_improvement(
    target: Tensor,
    estimate: Tensor,
    mixture: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Computes SI-SDR Improvement (SI-SDRi) relative to the unprocessed mixture.

    Formula:
        SI-SDRi = SI-SDR(target, estimate) - SI-SDR(target, mixture)
    """
    si_sdr_est = compute_si_sdr(target, estimate, eps=eps)
    si_sdr_mix = compute_si_sdr(target, mixture, eps=eps)
    return si_sdr_est - si_sdr_mix


# ===========================================================================
# 2. Multi-Resolution Spectral Losses
# ===========================================================================


class SpectralConvergenceLoss(nn.Module):
    """Frobenius-norm spectral convergence between predicted and target magnitudes."""

    def __init__(self, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        diff = torch.norm(target_mag - pred_mag, p="fro", dim=(-2, -1))
        norm = torch.clamp(torch.norm(target_mag, p="fro", dim=(-2, -1)), min=self.eps)
        sc = torch.clamp(diff / norm, max=20.0)
        return torch.mean(sc)


class LogSTFTMagnitudeLoss(nn.Module):
    """L1 distance between log-magnitude spectrograms."""

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_mag: Tensor, target_mag: Tensor) -> Tensor:
        log_pred = torch.log(torch.clamp(pred_mag, min=self.eps))
        log_target = torch.log(torch.clamp(target_mag, min=self.eps))
        return F.l1_loss(log_pred, log_target)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss for raw waveform source separation.

    Evaluates time-frequency spectral alignment at multiple window lengths
    (e.g., 512, 1024, 2048) to capture both transient attacks and fine harmonic pitch.
    """

    def __init__(
        self,
        fft_sizes: Tuple[int, ...] = (512, 1024, 2048),
        hop_sizes: Tuple[int, ...] = (128, 256, 512),
        win_lengths: Tuple[int, ...] = (512, 1024, 2048),
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
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
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


# ===========================================================================
# 3. Human Listener Playback & Multi-Track Audio Audit Utilities
# ===========================================================================


def display_audio_player(
    audio: Union[Tensor, np.ndarray],
    sample_rate: int = 44100,
    title: str = "Audio Track",
) -> Any:
    """Renders a playable audio widget inside Jupyter / Google Colab for a human listener."""
    try:
        from IPython.display import Audio, display

        if isinstance(audio, Tensor):
            audio = audio.detach().cpu().squeeze().numpy()
        elif isinstance(audio, np.ndarray):
            audio = np.squeeze(audio)

        # Normalize audio for comfortable playback
        peak = np.max(np.abs(audio))
        if peak > 1e-4:
            audio = audio / peak * 0.9

        duration_sec = audio.shape[-1] / sample_rate if audio.ndim > 1 else len(audio) / sample_rate
        print(f"🎧 [Player] {title} ({sample_rate} Hz, {duration_sec:.2f}s):")
        player = Audio(audio, rate=sample_rate)
        display(player)
        return player
    except Exception as e:
        print(f"Could not render IPython audio player: {e}")
        return None


def create_separation_audio_bundle(
    mixture: Tensor,
    target: Tensor,
    estimate: Tensor,
    sample_rate: int = 44100,
    sample_idx: int = 0,
) -> Dict[str, Any]:
    """Prepares and displays a multi-track comparison bundle for a human listener."""
    m = mixture[sample_idx].detach().cpu()
    t = target[sample_idx].detach().cpu()
    e = estimate[sample_idx].detach().cpu()
    err = t - e

    snr_val = float(compute_snr(t.unsqueeze(0), e.unsqueeze(0)).item())
    si_sdr_val = float(compute_si_sdr(t.unsqueeze(0), e.unsqueeze(0)).item())

    print("=" * 65)
    print("  Single Source Separation: Human Listener Audit")
    print(f"  SNR: {snr_val:+.2f} dB | SI-SDR: {si_sdr_val:+.2f} dB")
    print("=" * 65)

    display_audio_player(m, sample_rate, "1. Composite Mixture (Input)")
    display_audio_player(t, sample_rate, "2. Ground Truth Target (Clean)")
    display_audio_player(e, sample_rate, "3. Model Reconstructed Target (Separated)")
    display_audio_player(err, sample_rate, "4. Separation Error (Target - Estimate)")

    return {
        "snr": snr_val,
        "si_sdr": si_sdr_val,
        "mixture": m,
        "target": t,
        "estimate": e,
        "error": err,
    }
