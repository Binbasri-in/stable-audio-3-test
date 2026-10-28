"""Base class and common contracts for audio source separation inference pipelines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class BaseAudioPipeline(ABC):
    """Abstract base class for audio separation inference workflows and solvers."""

    def __init__(self, device: torch.device | str | None = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    @abstractmethod
    def separate(self, mixture: Tensor, **kwargs: Any) -> dict[str, Any]:
        """Separates an input mixture tensor into estimated source stems."""
        raise NotImplementedError

    def to(self, device: torch.device | str) -> BaseAudioPipeline:
        """Moves pipeline models to the specified device."""
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        return self


__all__ = ["BaseAudioPipeline"]
