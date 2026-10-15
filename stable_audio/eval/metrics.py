"""Evaluation metrics for audio generation and conditioned audio generation.

Implements:
- Fréchet Audio Distance (FAD): FAD_full, FAD_inpaint
- CLAP Scores: CLAP_text-gen, CLAP_gen-orig
- Latency, Throughput, and Real-Time Factor (RTF) Profiling
"""

import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import scipy.linalg
import torch


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculates Fréchet Distance between two multivariate Gaussians:
    FAD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * (sigma1 * sigma2)^(1/2))
    """
    diff = mu1 - mu2

    # Product of covariances
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical imaginary artifact removal
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # keep real part if imaginary component is negligible
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fad = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return max(0.0, fad)


def compute_fad(
    ref_embeddings: Union[np.ndarray, torch.Tensor],
    gen_embeddings: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-6,
) -> float:
    """Computes Fréchet Audio Distance (FAD) between reference and generated embeddings.
    
    Args:
        ref_embeddings: (N_ref, D) matrix of audio embeddings (e.g. from LAION-CLAP 630k).
        gen_embeddings: (N_gen, D) matrix of generated audio embeddings.
        eps: Regularization epsilon for numerical stability.

    Returns:
        Scalar FAD value (lower indicates higher distribution similarity).
    """
    if isinstance(ref_embeddings, torch.Tensor):
        ref_embeddings = ref_embeddings.detach().cpu().numpy()
    if isinstance(gen_embeddings, torch.Tensor):
        gen_embeddings = gen_embeddings.detach().cpu().numpy()

    assert ref_embeddings.ndim == 2, f"Expected 2D array, got {ref_embeddings.shape}"
    assert gen_embeddings.ndim == 2, f"Expected 2D array, got {gen_embeddings.shape}"
    assert ref_embeddings.shape[1] == gen_embeddings.shape[1], "Embedding dimension mismatch"

    mu_ref = np.mean(ref_embeddings, axis=0)
    sigma_ref = np.cov(ref_embeddings, rowvar=False) + np.eye(ref_embeddings.shape[1]) * eps

    mu_gen = np.mean(gen_embeddings, axis=0)
    sigma_gen = np.cov(gen_embeddings, rowvar=False) + np.eye(gen_embeddings.shape[1]) * eps

    return calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen, eps=eps)


def compute_clap_score(
    text_embeddings: Union[np.ndarray, torch.Tensor],
    audio_embeddings: Union[np.ndarray, torch.Tensor],
    pairwise: bool = True,
) -> Union[float, np.ndarray]:
    """Computes cosine similarity (CLAP score) between text and audio embeddings.
    
    Args:
        text_embeddings: (B, D) or (D,) text embeddings.
        audio_embeddings: (B, D) or (D,) audio embeddings.
        pairwise: If True, computes elementwise similarity for paired (B, D) arrays.
                  If False, returns full (B_text, B_audio) similarity matrix.

    Returns:
        Cosine similarity score(s) in [-1.0, 1.0]. Higher is better.
    """
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu().numpy()
    if isinstance(audio_embeddings, torch.Tensor):
        audio_embeddings = audio_embeddings.detach().cpu().numpy()

    if text_embeddings.ndim == 1:
        text_embeddings = text_embeddings[None, :]
    if audio_embeddings.ndim == 1:
        audio_embeddings = audio_embeddings[None, :]

    # Normalize vectors
    text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-8)
    audio_norm = audio_embeddings / (np.linalg.norm(audio_embeddings, axis=-1, keepdims=True) + 1e-8)

    if pairwise:
        assert text_norm.shape[0] == audio_norm.shape[0], "Batch sizes must match for pairwise scoring"
        sims = np.sum(text_norm * audio_norm, axis=-1)
        return float(np.mean(sims)) if len(sims) == 1 else sims
    else:
        return text_norm @ audio_norm.T


def compute_inpaint_metrics(
    original_audio_embeddings: np.ndarray,
    generated_audio_embeddings: np.ndarray,
    original_inpaint_embeddings: np.ndarray,
    generated_inpaint_embeddings: np.ndarray,
    text_prompt_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes the complete suite of inpainting evaluation metrics:
    
    1. FAD_full: FAD between (full) generated audio and (full) original reference audio.
       Assesses overall seamlessness, integration, and boundary transitions.
    2. FAD_inpaint: FAD considering only the generated (inpainted) regions and their
       corresponding original segments. Assesses quality of the generated part alone.
    3. CLAP_text-gen: Semantic similarity between prompt and generated inpainted region.
    4. CLAP_gen-orig: Perceptual acoustic similarity between generated inpainted segment
       and original segment.
    """
    fad_full = compute_fad(original_audio_embeddings, generated_audio_embeddings)
    fad_inpaint = compute_fad(original_inpaint_embeddings, generated_inpaint_embeddings)

    clap_gen_orig = float(np.mean(compute_clap_score(generated_inpaint_embeddings, original_inpaint_embeddings, pairwise=True)))

    clap_text_gen = 0.0
    if text_prompt_embeddings is not None:
        clap_text_gen = float(np.mean(compute_clap_score(text_prompt_embeddings, generated_inpaint_embeddings, pairwise=True)))

    return {
        "fad_full": fad_full,
        "fad_inpaint": fad_inpaint,
        "clap_text_gen": clap_text_gen,
        "clap_gen_orig": clap_gen_orig,
    }


class BenchmarkTimer:
    """High-precision latency, throughput, and Real-Time Factor (RTF) profiler."""

    def __init__(self, target_hardware: str = "Standard Hardware (e.g. H200 / MPS / CUDA)"):
        self.target_hardware = target_hardware
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_seconds: float = 0.0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        self.end_time = time.perf_counter()
        self.elapsed_seconds = self.end_time - (self.start_time or self.end_time)
        return self.elapsed_seconds

    def compute_rtf(self, audio_duration_seconds: float) -> float:
        """Real-Time Factor: latency / audio_duration. RTF < 1.0 indicates faster than real time."""
        if audio_duration_seconds <= 0:
            return 0.0
        return self.elapsed_seconds / audio_duration_seconds

    def summary(self, audio_duration_seconds: float, num_steps: int = 50) -> Dict[str, Union[float, str]]:
        rtf = self.compute_rtf(audio_duration_seconds)
        step_ms = (self.elapsed_seconds / num_steps) * 1000 if num_steps > 0 else 0.0
        return {
            "hardware": self.target_hardware,
            "latency_seconds": round(self.elapsed_seconds, 3),
            "audio_duration_seconds": round(audio_duration_seconds, 2),
            "real_time_factor": round(rtf, 4),
            "step_time_ms": round(step_ms, 2),
            "is_faster_than_realtime": rtf < 1.0,
        }
