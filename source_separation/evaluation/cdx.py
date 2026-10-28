"""Sound Demixing Challenge 2023 - Cinematic Demixing Track (SDX'23 CDX) Evaluation Suite.

Incorporates the formal setups, mathematical formulations, and evaluation protocols from:
"The Sound Demixing Challenge 2023 – Cinematic Demixing Track" (SDX'23 CDX)
Stefan Uhlich, Giorgio Fabbro, Masato Hirano, Shusuke Takahashi, Gordon Wichern,
Jonathan Le Roux, et al. (arXiv:2308.06981)

Key Features:
1. Task Formulation & Ranking Metric:
   - Stereo cinematic audio: x(n) = s_DX(n) + s_FX(n) + s_MX(n) at 44.1 kHz
   - Stems: Dialogue (DX), Sound Effects (FX), Music (MX)
   - Ranking Metric: Utterance-level Global SDR: SDR = 1/3 * (SDR_DX + SDR_FX + SDR_MX)
   - SDR_j = 10 * log10( sum_n ||s_j(n)||^2 / (sum_n ||s_j(n) - s_hat_j(n)||^2 + eps) )
2. Negative Anchor Baseline:
   - Scaled Identity Baseline: s_hat_j(n) = 1/3 * x(n)
3. Cocktail-Fork Baseline Setup & Post-Processing:
   - Peak Normalization: x(n) <- x(n) / max |x(n)|
   - Least-Squares Scale Estimation: alpha = <x, s_hat> / (||s_hat||^2 + eps)
   - Mixture Consistency Post-Processing (Wisdom et al. / Uhlich et al.):
     residual r = x - sum(s_hat); s_hat_FX += 0.5*r, s_hat_MX += 0.5*r
4. mp3d Enhancements:
   - ITU-R BS.1770-4 LUFS measurement and normalization (-27 LUFS / -15 LUFS)
   - Post-processing source scaling (1 / alpha on FX)
   - STFT Wiener filter refinement on Dialogue
5. Winning Cascaded Dialogue-First Demixing Pipeline:
   - Dialogue extraction -> subtraction -> 2-stem FX/MX separation
6. Quality Control & Signal Statistics:
   - Empirical RMS activity gating thresholds (tau_DX=0.022, tau_FX=0.005, tau_MX=0.003)
   - Stereo Panning Spectrum Psi(f), Delta(f)
   - 101-tap linear-phase FIR equalization matching filter
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import signal
from torch import Tensor, nn

from .common import HAS_PYLOUDNORM, pyln, to_numpy_channels_first
from .metrics import display_audio_player

# ===========================================================================
# 1. SDX 2023 Cinematic Demixing Track Metrics & Anchor
# ===========================================================================


def compute_utterance_sdr(
    target: Tensor | np.ndarray,
    estimate: Tensor | np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Computes utterance-level Signal-to-Distortion Ratio (SDR) (Paper 2, Eq. 2).

    Formula:
        SDR_j = 10 * log10( sum_n ||s_j(n)||^2 / (sum_n ||s_j(n) - s_hat_j(n)||^2 + eps) )

    where s_j(n) in R^2 and s_hat_j(n) in R^2 denote the stereo target and estimate.
    The norm ||.||^2 sums across channels and time samples.
    Equivalent to multichannel BSS Eval v3 SDR with zero filter length / scale.

    Args:
        target: Reference stereo clean audio [2, N] or batched.
        estimate: Model separated stereo audio [2, N] or batched.
        eps: Regularization constant (default: 1e-7).

    Returns:
        SDR in decibels (dB) as a float.
    """
    if isinstance(target, Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(estimate, Tensor):
        estimate = estimate.detach().cpu().numpy()

    target = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    # Align length if minor padding difference
    min_len = min(target.shape[-1], estimate.shape[-1])
    target = target[..., :min_len]
    estimate = estimate[..., :min_len]

    numerator = np.sum(target ** 2)
    denominator = np.sum((target - estimate) ** 2) + eps

    sdr = 10.0 * np.log10(max(numerator, eps) / denominator)
    return float(sdr)


def compute_cdx_global_sdr(
    targets: dict[str, Tensor | np.ndarray],
    estimates: dict[str, Tensor | np.ndarray],
    eps: float = 1e-7,
) -> dict[str, float]:
    """Computes the official CDX ranking metric: Global SDR (Paper 2, Section 2.3, Eq. 2).

    Formula:
        SDR = 1/3 * (SDR_DX + SDR_FX + SDR_MX)

    where DX = Dialogue, FX = Sound Effects, MX = Music.

    Args:
        targets: Dictionary containing ground truth stereo stems ('dialogue', 'effects', 'music')
                 or keys ('dx', 'fx', 'mx').
        estimates: Dictionary containing model separated stereo stems.
        eps: Regularization constant.

    Returns:
        Dictionary with:
            - 'global_sdr': mean SDR across the 3 stems in dB
            - 'sdr_dialogue': dialogue SDR in dB
            - 'sdr_effects': sound effects SDR in dB
            - 'sdr_music': music SDR in dB
    """
    canonical_keys = {
        "dx": "dialogue",
        "speech": "dialogue",
        "dialogue": "dialogue",
        "fx": "effects",
        "sfx": "effects",
        "effects": "effects",
        "mx": "music",
        "music": "music",
    }

    t_map = {canonical_keys[k.lower()]: v for k, v in targets.items() if k.lower() in canonical_keys}
    e_map = {canonical_keys[k.lower()]: v for k, v in estimates.items() if k.lower() in canonical_keys}

    required = ["dialogue", "effects", "music"]
    stem_sdrs: dict[str, float] = {}

    for stem in required:
        if stem in t_map and stem in e_map:
            stem_sdrs[stem] = compute_utterance_sdr(t_map[stem], e_map[stem], eps=eps)
        else:
            stem_sdrs[stem] = float("-inf")

    global_sdr = float(np.mean([stem_sdrs[s] for s in required]))

    return {
        "global_sdr": global_sdr,
        "sdr_dialogue": stem_sdrs["dialogue"],
        "sdr_effects": stem_sdrs["effects"],
        "sdr_music": stem_sdrs["music"],
    }


def compute_cdx_scaled_identity_baseline(
    mixture: Tensor | np.ndarray,
    targets: dict[str, Tensor | np.ndarray] | None = None,
) -> dict[str, Any]:
    """Computes the Scaled Identity negative anchor baseline (Paper 2, Section 4, Table 1).

    Formula:
        s_hat_j(n) = 1/3 * x(n)   for j in {DX, FX, MX}

    In CDXDB23, this yields -0.019 dB mean SDR (Dialogue: +1.562 dB, FX: -1.236 dB, MX: -0.383 dB).
    """
    mix_np = to_numpy_channels_first(mixture)
    scaled_mix = (1.0 / 3.0) * mix_np

    estimates = {
        "dialogue": scaled_mix.copy(),
        "effects": scaled_mix.copy(),
        "music": scaled_mix.copy(),
    }

    out: dict[str, Any] = {"estimates": estimates}
    if targets is not None:
        out["scores"] = compute_cdx_global_sdr(targets, estimates)
    return out


# ===========================================================================
# 2. Cocktail-Fork & mp3d Processing Pipelines
# ===========================================================================


def peak_normalize_audio(
    audio: Tensor | np.ndarray,
    eps: float = 1e-8,
) -> tuple[Tensor | np.ndarray, float]:
    """Applies input peak normalization (Paper 2, Section 3.3, Eq. 3).

    Formula:
        x(n) <- x(n) / max_n |x(n)|

    Returns:
        (normalized_audio, peak_scale_factor)
    """
    if isinstance(audio, Tensor):
        peak = torch.max(torch.abs(audio)).item()
        scale = 1.0 / max(peak, eps)
        return audio * scale, scale
    else:
        audio_np = np.asarray(audio)
        peak = float(np.max(np.abs(audio_np)))
        scale = 1.0 / max(peak, eps)
        return audio_np * scale, scale


def least_squares_scale_estimation(
    mixture: Tensor | np.ndarray,
    estimate: Tensor | np.ndarray,
    eps: float = 1e-7,
) -> tuple[Tensor | np.ndarray, float]:
    """Recovers optimal scale using least-squares matching to mixture (Paper 2, Eq. 4).

    When models are trained with scale-invariant losses (like SI-SDR loss), scale needs
    to be estimated according to:
        s_hat_j(n) <- [ sum_n x(n)^T s_hat_j(n) / (10^-7 + sum_n ||s_hat_j(n)||^2) ] * s_hat_j(n)

    Args:
        mixture: Stereo or multichannel mixture x(n) [C, T].
        estimate: Model separated stem s_hat_j(n) [C, T].
        eps: Regularization constant (1e-7 as in Eq. 4).

    Returns:
        (scaled_estimate, alpha)
    """
    is_tensor = isinstance(estimate, Tensor)
    if is_tensor:
        m = mixture.detach()
        e = estimate.detach()
        dot_product = torch.sum(m * e).item()
        energy = torch.sum(e ** 2).item() + eps
        alpha = dot_product / energy
        return estimate * alpha, alpha
    else:
        m = np.asarray(mixture)
        e = np.asarray(estimate)
        dot_product = float(np.sum(m * e))
        energy = float(np.sum(e ** 2)) + eps
        alpha = dot_product / energy
        return e * alpha, alpha


def apply_mixture_consistency(
    mixture: Tensor | np.ndarray,
    estimates: dict[str, Tensor | np.ndarray],
    mode: str = "cocktail_fork",
    custom_weights: dict[str, float] | None = None,
) -> dict[str, Tensor | np.ndarray]:
    """Applies mixture consistency post-processing (Wisdom et al., 2019; Paper 2, Section 3.3).

    Computes residual:
        r(n) = x(n) - sum_j s_hat_j(n)

    Modes:
    1. 'cocktail_fork' / 'music_sfx' (Wisdom et al. / Petermann et al. / Uhlich et al.):
       Residual contains mostly sound effects and background music; dialogue has minimal residual.
       s_hat_DX(n) <- s_hat_DX(n)
       s_hat_FX(n) <- s_hat_FX(n) + 0.5 * r(n)
       s_hat_MX(n) <- s_hat_MX(n) + 0.5 * r(n)
       Yields +1.1 dB on SDR_FX, +0.7 dB on SDR_MX, and +0.6 dB overall SDR improvement!

    2. 'equal': Distributes residual equally across all estimated stems:
       s_hat_j(n) <- s_hat_j(n) + (1 / J) * r(n)

    3. 'custom': Uses custom_weights dict to distribute residual.
    """
    is_tensor = any(isinstance(v, Tensor) for v in estimates.values())
    stem_keys = list(estimates.keys())

    if is_tensor:
        m = mixture if isinstance(mixture, Tensor) else torch.from_numpy(mixture)
        est_tensors = {k: (v if isinstance(v, Tensor) else torch.from_numpy(v)) for k, v in estimates.items()}
        sum_est = sum(est_tensors.values())
        res = m - sum_est

        refined: dict[str, Tensor] = {}
        if mode in ("cocktail_fork", "music_sfx"):
            for k in stem_keys:
                k_lower = k.lower()
                if "dialogue" in k_lower or "speech" in k_lower or "dx" in k_lower:
                    refined[k] = est_tensors[k]
                else:
                    refined[k] = est_tensors[k] + 0.5 * res
        elif mode == "equal":
            weight = 1.0 / len(stem_keys)
            for k in stem_keys:
                refined[k] = est_tensors[k] + weight * res
        elif mode == "custom" and custom_weights is not None:
            total_w = sum(custom_weights.values()) + 1e-8
            for k in stem_keys:
                w = custom_weights.get(k, 0.0) / total_w
                refined[k] = est_tensors[k] + w * res
        else:
            refined = est_tensors
        return refined
    else:
        m = np.asarray(mixture)
        est_arrays = {k: np.asarray(v) for k, v in estimates.items()}
        sum_est = sum(est_arrays.values())
        res = m - sum_est

        refined_np: dict[str, np.ndarray] = {}
        if mode in ("cocktail_fork", "music_sfx"):
            for k in stem_keys:
                k_lower = k.lower()
                if "dialogue" in k_lower or "speech" in k_lower or "dx" in k_lower:
                    refined_np[k] = est_arrays[k].copy()
                else:
                    refined_np[k] = est_arrays[k] + 0.5 * res
        elif mode == "equal":
            weight = 1.0 / len(stem_keys)
            for k in stem_keys:
                refined_np[k] = est_arrays[k] + weight * res
        elif mode == "custom" and custom_weights is not None:
            total_w = sum(custom_weights.values()) + 1e-8
            for k in stem_keys:
                w = custom_weights.get(k, 0.0) / total_w
                refined_np[k] = est_arrays[k] + w * res
        else:
            refined_np = est_arrays
        return refined_np


def measure_loudness_lufs(
    audio: Tensor | np.ndarray,
    sample_rate: int = 44100,
) -> float:
    """Measures integrated loudness in LUFS following ITU-R BS.1770-4 (Paper 2, Section 4.3 & 5.1).

    Uses `pyloudnorm.Meter` if installed, or standard K-weighting pre-filter implementation.
    """
    audio_np = to_numpy_channels_first(audio).T  # [T, C]

    if HAS_PYLOUDNORM:
        try:
            meter = pyln.Meter(sample_rate)
            return float(meter.integrated_loudness(audio_np))
        except Exception:
            pass

    # Built-in fallback: Standard ITU-R BS.1770-4 K-weighting filter
    b1, a1 = signal.bilinear([1.0, 1.5, 1.0], [1.0, 1.0, 1.0], fs=sample_rate)
    filtered = signal.lfilter(b1, a1, audio_np, axis=0)
    ms = np.mean(filtered ** 2, axis=0)
    lufs = -0.691 + 10.0 * np.log10(np.sum(ms) + 1e-10)
    return float(lufs)


def normalize_loudness_lufs(
    audio: Tensor | np.ndarray,
    target_lufs: float = -27.0,
    sample_rate: int = 44100,
) -> tuple[Tensor | np.ndarray, float]:
    """Normalizes audio to a target integrated LUFS level (Paper 2, Section 4.3.2 & 5.2).

    Cocktail-fork baseline v1.1 and Team mp3d found input volume normalization crucial:
    - -27.0 LUFS: Optimal for models trained on Divide and Remaster (DnR).
    - -15.0 LUFS: Optimal for real movie soundtracks / RED dataset.

    Returns:
        (normalized_audio, linear_gain)
    """
    is_tensor = isinstance(audio, Tensor)
    current_lufs = measure_loudness_lufs(audio, sample_rate=sample_rate)
    gain_db = target_lufs - current_lufs
    gain = 10.0 ** (gain_db / 20.0)

    if is_tensor:
        return audio * gain, gain
    else:
        return np.asarray(audio) * gain, gain


def apply_post_processing_scaling(
    mixture: Tensor | np.ndarray,
    estimate: Tensor | np.ndarray,
    eps: float = 1e-7,
) -> Tensor | np.ndarray:
    """Applies Team mp3d's post-processing source scaling (Paper 2, Section 4.3.2).

    Multiplies estimated source by 1 / alpha, where:
        alpha = sum_n x(n)^T s_hat(n) / (10^-7 + sum_n ||s_hat(n)||^2)

    mp3d verified that applying this scaling to the Sound Effects stem yielded +0.3 dB SDR improvement.
    """
    _, alpha = least_squares_scale_estimation(mixture, estimate, eps=eps)
    scale = 1.0 / max(alpha, 1e-4)
    return estimate * scale


def apply_wiener_filter_postprocessing(
    estimates: dict[str, Tensor | np.ndarray],
    mixture: Tensor | np.ndarray,
    target_stems_for_wiener: list[str] | None = None,
    n_fft: int = 2048,
    hop_length: int | None = None,
    eps: float = 1e-7,
) -> dict[str, Tensor | np.ndarray]:
    """Applies STFT-based Wiener filter refinement to separated stems (Paper 2, Section 4.3.2).

    Team mp3d demonstrated that applying a Wiener filter to dialogue improves Dialogue SDR by +0.3 dB.
    """
    if hop_length is None:
        hop_length = n_fft // 2

    if target_stems_for_wiener is None:
        target_stems_for_wiener = ["dialogue", "dx", "speech"]

    mix_np = to_numpy_channels_first(mixture)
    n_samples = mix_np.shape[-1]
    _, _, X = signal.stft(mix_np, nperseg=n_fft, noverlap=n_fft - hop_length)

    P: dict[str, np.ndarray] = {}
    total_power = np.full_like(np.abs(X), eps)

    for k, v in estimates.items():
        v_np = to_numpy_channels_first(v)
        _, _, Yk = signal.stft(v_np, nperseg=n_fft, noverlap=n_fft - hop_length)
        P[k] = np.abs(Yk) ** 2
        total_power += P[k]

    refined: dict[str, Any] = {}
    for k, v in estimates.items():
        is_target = any(t in k.lower() for t in target_stems_for_wiener)
        if is_target:
            mask = P[k] / total_power
            Y_filt = X * mask
            _, est_audio = signal.istft(Y_filt, nperseg=n_fft, noverlap=n_fft - hop_length)
            est_audio = est_audio[:, :n_samples]
            refined[k] = torch.from_numpy(est_audio).to(v.device) if isinstance(v, Tensor) else est_audio
        else:
            refined[k] = v

    return refined


# ===========================================================================
# 3. Cascaded Dialogue-First Demixing Pipeline (JusperLee / ZFTurbo)
# ===========================================================================


class CascadedCinematicDemixer(nn.Module):
    """Dialogue-first cascaded cinematic demixing architecture (Paper 2, Section 4.1 & 4.2).

    Top winning strategies on Leaderboard B (JusperLee 1st place, ZFTurbo 2nd place):
    1. Separate Dialogue first using a specialized vocal/speech extraction model:
       s_hat_DX = Model_DX(x)
    2. Subtract dialogue from the mixture:
       x_rem = x - s_hat_DX
    3. Separate the remaining non-dialogue mixture x_rem into Effects and Music using a 2-stem model:
       (s_hat_FX, s_hat_MX) = Model_FX_MX(x_rem)

    Simplifying the remaining mixture without speech improves music SDR by over 2.5 dB!
    """

    def __init__(
        self,
        dialogue_model: nn.Module,
        background_model: nn.Module,
        apply_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.dialogue_model = dialogue_model
        self.background_model = background_model
        self.apply_consistency = apply_consistency

    def forward(self, mixture: Tensor) -> dict[str, Tensor]:
        """Separates stereo mixture [B, 2, T] into dialogue, effects, and music stems."""
        # 1. Dialogue extraction
        dx_out = self.dialogue_model(mixture)
        s_hat_dx = dx_out["recon"] if isinstance(dx_out, dict) else (dx_out[0] if isinstance(dx_out, (list, tuple)) else dx_out)

        # 2. Subtract dialogue to create pseudo music-effects mixture
        mixture_rem = mixture - s_hat_dx

        # 3. Two-stem background separation (FX and MX)
        bg_out = self.background_model(mixture_rem)
        if isinstance(bg_out, dict):
            s_hat_fx = bg_out.get("effects", bg_out.get("fx", bg_out.get("sfx")))
            s_hat_mx = bg_out.get("music", bg_out.get("mx"))
            if s_hat_fx is None or s_hat_mx is None:
                recon = bg_out.get("recon", 0.5 * mixture_rem)
                s_hat_fx = 0.5 * recon
                s_hat_mx = 0.5 * recon
        elif isinstance(bg_out, (list, tuple)) and len(bg_out) >= 2:
            s_hat_fx, s_hat_mx = bg_out[0], bg_out[1]
        else:
            s_hat_fx = 0.5 * mixture_rem
            s_hat_mx = 0.5 * mixture_rem

        estimates = {
            "dialogue": s_hat_dx,
            "effects": s_hat_fx,
            "music": s_hat_mx,
        }

        # 4. Optional cocktail-fork mixture consistency
        if self.apply_consistency:
            estimates = apply_mixture_consistency(mixture, estimates, mode="cocktail_fork")

        return estimates


# ===========================================================================
# 4. Quality Control, Activity Gating & Signal Statistics Analysis
# ===========================================================================


def check_audio_activity_rms(
    audio: Tensor | np.ndarray,
    stem_type: str = "dialogue",
    custom_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluates RMS activity against the CDXDB23 empirical thresholds (Paper 2, Section 3.2).

    Empirical activity thresholds defined in Paper 2:
        tau_DX = 0.022  (Dialogue)
        tau_FX = 0.005  (Sound Effects)
        tau_MX = 0.003  (Music)

    Clips with RMS below threshold are considered inactive / unvoiced.

    Formula:
        RMS_j = sqrt( (1 / N) * sum_{n=1}^N ||s_j(n)||^2 )

    Returns:
        {'rms': float, 'threshold': float, 'is_active': bool}
    """
    audio_np = to_numpy_channels_first(audio)
    rms = float(np.sqrt(np.mean(np.sum(audio_np ** 2, axis=0))))

    thresholds = {
        "dialogue": 0.022,
        "dx": 0.022,
        "speech": 0.022,
        "effects": 0.005,
        "fx": 0.005,
        "sfx": 0.005,
        "music": 0.003,
        "mx": 0.003,
    }

    stem_lower = stem_type.lower()
    thresh = custom_threshold if custom_threshold is not None else thresholds.get(stem_lower, 0.005)

    return {
        "rms": rms,
        "threshold": thresh,
        "is_active": rms >= thresh,
    }


def compute_stereo_panning_spectrum(
    audio: Tensor | np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 4096,
    overlap_ratio: float = 0.75,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Computes Stereo Panning Spectrum Psi(f) and Delta(f) (Paper 2, Section 5.1, Eq. 5a & 5b).

    Tzanetakis et al. (2007) and Avendano (2003):
        Psi(f) = 2 * (X_L(f) * X_R(f)) / (X_L(f)^2 + X_R(f)^2)
        Delta(f) = sign( (X_L(f) * X_R(f))/X_L(f)^2 - (X_L(f) * X_R(f))/X_R(f)^2 )

    - Psi(f) in [0, 1] measures channel amplitude similarity (1 = centered, <1 = panned).
    - Delta(f) in {-1, 0, 1} measures direction (-1 = left, +1 = right).
    """
    audio_np = to_numpy_channels_first(audio)
    if audio_np.shape[0] < 2:
        raise ValueError("Stereo audio required (expected 2 channels).")

    hop = int(n_fft * (1.0 - overlap_ratio))
    f, _, Zxx = signal.stft(audio_np[:2], fs=sample_rate, window="hann", nperseg=n_fft, noverlap=n_fft - hop)

    XL = np.mean(np.abs(Zxx[0]), axis=-1)
    XR = np.mean(np.abs(Zxx[1]), axis=-1)

    psi = 2.0 * (XL * XR) / (XL ** 2 + XR ** 2 + eps)
    term_l = (XL * XR) / (XL ** 2 + eps)
    term_r = (XL * XR) / (XR ** 2 + eps)
    delta = np.sign(term_l - term_r)

    return {
        "frequencies": f,
        "psi": psi,
        "delta": delta,
    }


def design_equalization_matching_filter(
    source_magnitude_curve: np.ndarray,
    target_magnitude_curve: np.ndarray,
    num_taps: int = 101,
) -> np.ndarray:
    """Designs a 101-tap linear-phase FIR filter to match equalization spectra (Paper 2, Section 5.2).

    The magnitude response of this filter is the square root of the ratio between
    target and source equalization. Applied using forward-backward filtering (`signal.filtfilt`).
    """
    ratio = target_magnitude_curve / (source_magnitude_curve + 1e-10)
    desired_gain = np.sqrt(np.clip(ratio, 1e-3, 100.0))

    freqs = np.linspace(0.0, 1.0, len(desired_gain))
    b = signal.firwin2(num_taps, freqs, desired_gain)
    return b


# ===========================================================================
# 5. High-Level CDX Track Evaluation Runner & Listener Audit
# ===========================================================================


def evaluate_cdx_cinematic_separation(
    mixture: Tensor | np.ndarray,
    targets: dict[str, Tensor | np.ndarray],
    estimates: dict[str, Tensor | np.ndarray],
    apply_peak_norm: bool = False,
    apply_ls_scaling: bool = True,
    apply_consistency: bool = True,
    lufs_norm_target: float | None = None,
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """Runs a complete CDX track evaluation pipeline (SDX'23 Cinematic Demixing Track).

    Includes:
    1. Optional input LUFS loudness / peak normalization
    2. Least-squares scale estimation on estimates (Paper 2, Eq. 4)
    3. Mixture consistency post-processing (Wisdom et al. / Paper 2, Section 3.3)
    4. Utterance-level Global SDR and stem SDRs (DX, FX, MX)
    5. Scaled Identity negative anchor comparison (1/3 * mixture)
    6. RMS activity gating verification (tau_DX=0.022, tau_FX=0.005, tau_MX=0.003)

    Returns:
        Structured evaluation metrics ready for publication and comparison tables.
    """
    mix_np = to_numpy_channels_first(mixture)
    t_map = {k.lower(): to_numpy_channels_first(v) for k, v in targets.items()}
    e_map = {k.lower(): to_numpy_channels_first(v) for k, v in estimates.items()}

    # 1. Preprocessing normalization
    if lufs_norm_target is not None:
        mix_np, _ = normalize_loudness_lufs(mix_np, target_lufs=lufs_norm_target, sample_rate=sample_rate)

    # 2. Scale estimation
    if apply_ls_scaling:
        for k in list(e_map.keys()):
            e_map[k], _ = least_squares_scale_estimation(mix_np, e_map[k])

    # 3. Mixture consistency
    if apply_consistency:
        e_map = apply_mixture_consistency(mix_np, e_map, mode="cocktail_fork")

    # 4. Global SDR
    scores = compute_cdx_global_sdr(t_map, e_map)

    # 5. Scaled Identity baseline comparison
    scaled_baseline = compute_cdx_scaled_identity_baseline(mix_np, t_map)

    # 6. Activity gating checks
    activity_status: dict[str, Any] = {}
    for stem in ["dialogue", "effects", "music"]:
        if stem in t_map:
            activity_status[stem] = check_audio_activity_rms(t_map[stem], stem_type=stem)

    return {
        "global_sdr": scores["global_sdr"],
        "sdr_dialogue": scores["sdr_dialogue"],
        "sdr_effects": scores["sdr_effects"],
        "sdr_music": scores["sdr_music"],
        "scaled_identity_scores": scaled_baseline.get("scores"),
        "activity_gating": activity_status,
        "refined_estimates": e_map,
    }


def create_cdx_separation_audit_bundle(
    mixture: Tensor | np.ndarray,
    targets: dict[str, Tensor | np.ndarray],
    estimates: dict[str, Tensor | np.ndarray],
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """Displays an interactive multi-track human listening audit for Cinematic Demixing (CDX)."""
    scores = compute_cdx_global_sdr(targets, estimates)
    print("=" * 70)
    print("  Cinematic Source Demixing (CDX SDX'23) Listener Audit")
    print(f"  Global SDR: {scores['global_sdr']:+.2f} dB")
    print(f"  Dialogue: {scores['sdr_dialogue']:+.2f} dB | Effects: {scores['sdr_effects']:+.2f} dB | Music: {scores['sdr_music']:+.2f} dB")
    print("=" * 70)

    display_audio_player(mixture, sample_rate, "Cinematic Input Mixture (Stereo)")
    for stem in ["dialogue", "effects", "music"]:
        t_key = next((k for k in targets if stem in k.lower()), None)
        e_key = next((k for k in estimates if stem in k.lower()), None)
        if t_key and e_key:
            sdr = compute_utterance_sdr(targets[t_key], estimates[e_key])
            display_audio_player(targets[t_key], sample_rate, f"Ground Truth: {stem.title()}")
            display_audio_player(estimates[e_key], sample_rate, f"Estimated: {stem.title()} (SDR: {sdr:+.2f} dB)")

    return scores
