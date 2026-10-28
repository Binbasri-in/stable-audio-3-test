"""Fundamental source separation metrics, multi-resolution losses, and audio audit tools."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor

from ..functional import (
    si_sdr as _si_sdr,
)
from ..functional import (
    si_sdr_improvement as _si_sdr_improvement,
)
from ..functional import (
    snr as _snr,
)
from ..losses import (
    LogSTFTMagnitudeLoss,
    MultiResolutionSTFTLoss,
    SpectralConvergenceLoss,
)

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
    return _snr(estimate, target, eps=eps)


def compute_si_sdr(
    target: Tensor,
    estimate: Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tensor:
    """Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).

    Args:
        target: Reference ground truth clean source [B, C, T] or [B, T].
        estimate: Model separated estimation [B, C, T] or [B, T].
        eps: Numerical stability constant.
        zero_mean: Whether to subtract sample mean before projection.

    Returns:
        [B] SI-SDR in dB for each sample in the batch.
    """
    return _si_sdr(estimate, target, eps=eps, zero_mean=zero_mean)


def compute_si_sdr_improvement(
    target: Tensor,
    estimate: Tensor,
    mixture: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Computes SI-SDR Improvement (SI-SDRi) relative to the unprocessed mixture."""
    return _si_sdr_improvement(estimate, target, mixture, eps=eps)


# ===========================================================================
# 2. Multi-Resolution Spectral Losses (Re-exported from source_separation.losses)
# ===========================================================================



# ===========================================================================
# 3. Human Listener Playback & Multi-Track Audio Audit Utilities
# ===========================================================================


def display_audio_player(
    audio: Tensor | np.ndarray,
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
) -> dict[str, Any]:
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


__all__ = [
    "LogSTFTMagnitudeLoss",
    "MultiResolutionSTFTLoss",
    "SpectralConvergenceLoss",
    "compute_si_sdr",
    "compute_si_sdr_improvement",
    "compute_snr",
    "create_separation_audio_bundle",
    "display_audio_player",
]
