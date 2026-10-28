"""Common helper functions and utilities for source separation evaluation."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

# Shared evaluation logger
logger = logging.getLogger("source_separation_eval")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Eval]: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Optional third-party packages with graceful fallbacks
try:
    import museval
    import museval.metrics
    HAS_MUSEVAL = True
except ImportError:
    museval = None
    HAS_MUSEVAL = False

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    pyln = None
    HAS_PYLOUDNORM = False


def to_numpy_channels_first(audio: Tensor | np.ndarray) -> np.ndarray:
    """Converts audio to a float32 numpy array with channels-first shape [C, T].

    Args:
        audio: Audio waveform as Tensor or ndarray with shape [T], [C, T], or [T, C].

    Returns:
        np.ndarray with shape [C, T] where C is number of channels (1 for mono, 2 for stereo).
    """
    if isinstance(audio, Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 1:
        audio = audio[None, :]  # [1, T]
    elif audio.ndim == 2 and audio.shape[0] > audio.shape[1] and audio.shape[1] <= 8:
        # Array passed as [T, C]
        audio = audio.T
    return audio


def to_tensor_channels_first(
    audio: Tensor | np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Converts audio to a PyTorch Tensor with channels-first shape [C, T] or [B, C, T].

    Args:
        audio: Audio waveform as Tensor or ndarray.
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        Tensor on specified device.
    """
    if isinstance(audio, np.ndarray):
        t = torch.from_numpy(audio).to(dtype=dtype)
    else:
        t = audio.to(dtype=dtype)

    if device is not None:
        t = t.to(device=device)

    if t.ndim == 1:
        t = t.unsqueeze(0)  # [1, T]
    elif t.ndim == 2 and t.shape[0] > t.shape[1] and t.shape[1] <= 8:
        t = t.transpose(0, 1)  # [C, T]
    return t


def invert_2x2(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Inverts 2x2 complex matrices M (batch of 2x2 matrices at trailing dimensions).

    Analytical inverse formula used in MWF oracle and spatial filtering.
    """
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    inv_det = 1.0 / (eps + det)
    inv_M = np.zeros(M.shape, dtype=np.complex64)
    inv_M[..., 0, 0] = inv_det * M[..., 1, 1]
    inv_M[..., 1, 0] = -inv_det * M[..., 1, 0]
    inv_M[..., 0, 1] = -inv_det * M[..., 0, 1]
    inv_M[..., 1, 1] = inv_det * M[..., 0, 0]
    return inv_M
