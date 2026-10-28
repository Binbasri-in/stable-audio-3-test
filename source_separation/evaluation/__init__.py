"""Audio Source Separation Evaluation and Training Validation Package.

Contains standardized benchmarks, oracle separation ceilings, post-processing pipelines,
and specialized periodic validation methods for audio source separation models:

Measurements & Submodules:
- `sisec`: Measurement 1 - The 2018 Signal Separation Evaluation Campaign (SiSEC 2018, Stöter et al.)
           BSS Eval Version 4 (SDR, ISR, SIR, SAR), IBM1, IBM2, IRM1, IRM2, MWF, and MIX anchor.
- `cdx`: Measurement 2 - The Sound Demixing Challenge 2023 Cinematic Demixing Track (SDX'23 CDX, Uhlich et al.)
         Utterance Global SDR (Dialogue, Effects, Music), Scaled Identity anchor, Cocktail-Fork post-processing,
         LUFS loudness normalization, RMS activity gating, and Cascaded Demixer.
- `metrics`: Fundamental quantitative metrics (SNR, SI-SDR, SI-SDRi) and multi-resolution STFT spectral losses.
- `validation`: Dedicated training validation methods (validate_model_during_training, validate_batch,
                SeparationTrainingValidator, and early-stopping manager).
"""

from __future__ import annotations

from .cdx import (
    CascadedCinematicDemixer,
    apply_mixture_consistency,
    apply_post_processing_scaling,
    apply_wiener_filter_postprocessing,
    check_audio_activity_rms,
    compute_cdx_global_sdr,
    compute_cdx_scaled_identity_baseline,
    compute_stereo_panning_spectrum,
    compute_utterance_sdr,
    create_cdx_separation_audit_bundle,
    design_equalization_matching_filter,
    evaluate_cdx_cinematic_separation,
    least_squares_scale_estimation,
    measure_loudness_lufs,
    normalize_loudness_lufs,
    peak_normalize_audio,
)
from .common import (
    invert_2x2,
    to_numpy_channels_first,
    to_tensor_channels_first,
)
from .metrics import (
    LogSTFTMagnitudeLoss,
    MultiResolutionSTFTLoss,
    SpectralConvergenceLoss,
    compute_si_sdr,
    compute_si_sdr_improvement,
    compute_snr,
    create_separation_audio_bundle,
    display_audio_player,
)
from .sisec import (
    compute_all_sisec_oracles,
    compute_bss_eval_v4,
    compute_mix_anchor,
    compute_oracle_ibm,
    compute_oracle_ibm1,
    compute_oracle_ibm2,
    compute_oracle_irm,
    compute_oracle_irm1,
    compute_oracle_irm2,
    compute_oracle_mwf,
    create_sisec_separation_audit_bundle,
    evaluate_sisec_mus18,
)
from .validation import (
    SeparationTrainingValidator,
    TrainingValidationResult,
    evaluate_source_separation,
    validate_batch,
    validate_model_during_training,
)

__all__ = [
    # 1. Fundamental Metrics & Losses
    "compute_snr",
    "compute_si_sdr",
    "compute_si_sdr_improvement",
    "SpectralConvergenceLoss",
    "LogSTFTMagnitudeLoss",
    "MultiResolutionSTFTLoss",
    "display_audio_player",
    "create_separation_audio_bundle",
    # 2. Measurement 1: SiSEC 2018 Oracles & BSS Eval v4
    "compute_oracle_ibm",
    "compute_oracle_ibm1",
    "compute_oracle_ibm2",
    "compute_oracle_irm",
    "compute_oracle_irm1",
    "compute_oracle_irm2",
    "compute_oracle_mwf",
    "compute_all_sisec_oracles",
    "compute_mix_anchor",
    "compute_bss_eval_v4",
    "evaluate_sisec_mus18",
    "create_sisec_separation_audit_bundle",
    # 3. Measurement 2: SDX'23 CDX Demixing Track
    "compute_utterance_sdr",
    "compute_cdx_global_sdr",
    "compute_cdx_scaled_identity_baseline",
    "peak_normalize_audio",
    "least_squares_scale_estimation",
    "apply_mixture_consistency",
    "measure_loudness_lufs",
    "normalize_loudness_lufs",
    "apply_post_processing_scaling",
    "apply_wiener_filter_postprocessing",
    "CascadedCinematicDemixer",
    "check_audio_activity_rms",
    "compute_stereo_panning_spectrum",
    "design_equalization_matching_filter",
    "evaluate_cdx_cinematic_separation",
    "create_cdx_separation_audit_bundle",
    # 4. Training Validation Methods
    "validate_model_during_training",
    "validate_batch",
    "SeparationTrainingValidator",
    "TrainingValidationResult",
    "evaluate_source_separation",
    # 5. Common Utilities
    "to_numpy_channels_first",
    "to_tensor_channels_first",
    "invert_2x2",
]
