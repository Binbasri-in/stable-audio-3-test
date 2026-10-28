"""Base Trainer and Shared Infrastructure for Audio Source Separation Trainers.

Provides a unified base class, logging configuration, device management,
and checkpoint serialization utilities shared across all specialized trainer modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

# Shared trainer logger
logger = logging.getLogger("source_separation_trainer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Trainer]: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def extract_reconstruction(model_output: Any) -> Tensor:
    """Standardizes model outputs from dict, tuple, list, or direct Tensor.

    Args:
        model_output: Output from a separation or reconstruction model.

    Returns:
        Tensor representing the predicted audio reconstruction waveform.
    """
    if isinstance(model_output, dict):
        if "recon" in model_output:
            return model_output["recon"]
        raise KeyError(f"Expected 'recon' key in output dict, found keys: {list(model_output.keys())}")
    elif isinstance(model_output, (tuple, list)):
        return model_output[0]
    elif isinstance(model_output, Tensor):
        return model_output
    raise TypeError(f"Unsupported model output type: {type(model_output)}")


class BaseTrainer:
    """Base class providing device resolution, checkpoint directories, and history tracking."""

    def __init__(
        self,
        device: torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.history: dict[str, list[float]] = {}

        # Hardware-aware Automatic Mixed Precision (AMP)
        device_type = self.device.type if self.device.type in ("cuda", "mps", "cpu") else "cpu"
        self.autocast_context = torch.amp.autocast(device_type=device_type, enabled=self.use_amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.device.type == "cuda"))

    def _resolve_device(self, module_or_tensor: Any | None = None) -> torch.device:
        """Resolves device from a given module or tensor if available, otherwise defaults to self.device."""
        if module_or_tensor is not None and isinstance(module_or_tensor, nn.Module):
            try:
                return next(module_or_tensor.parameters()).device
            except StopIteration:
                pass
        return self.device


__all__ = [
    "BaseTrainer",
    "extract_reconstruction",
    "logger",
]
