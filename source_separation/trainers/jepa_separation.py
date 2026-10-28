"""JEPA-Guided Separation Head Trainer.

Provides downstream supervised training for JEPAGuidedSeparationHead conditioned
on frozen Audio-JEPA acoustic priors using Multi-Resolution STFT and L1 reconstruction losses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseTrainer, logger

try:
    from ..evaluation import MultiResolutionSTFTLoss
except (ImportError, ValueError):
    try:
        from source_separation.evaluation import MultiResolutionSTFTLoss
    except ImportError:
        from evaluation import MultiResolutionSTFTLoss


class JEPASeparationTrainer(BaseTrainer):
    """Trainer for JEPAGuidedSeparationHead conditioned on frozen Audio-JEPA priors."""

    def __init__(
        self,
        sep_head: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        max_grad_norm: float = 1.0,
        use_amp: bool = False,
    ) -> None:
        resolved_device = device or self._resolve_device(sep_head)
        super().__init__(
            device=resolved_device,
            checkpoint_dir=checkpoint_dir,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
        )

        self.sep_head = sep_head.to(self.device)
        self.optimizer = optimizer
        self.stft_loss_fn = MultiResolutionSTFTLoss().to(self.device)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "l1_loss": [],
            "stft_loss": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Runs one epoch of downstream separation training."""
        self.sep_head.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_stft = 0.0
        num_batches = 0

        for batch in dataloader:
            mix = batch["mix_audio"].to(self.device)
            target = batch["target_audio"].to(self.device)

            self.optimizer.zero_grad()
            with self.autocast_context:
                # Separate using target as prior reference
                pred = self.sep_head(mix, clean_source_ref=target)

                l1 = F.l1_loss(pred, target)
                stft = self.stft_loss_fn(pred, target)
                loss = l1 + 0.5 * stft

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.sep_head.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.sep_head.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += float(loss.item())
            total_l1 += float(l1.item())
            total_stft += float(stft.item())
            num_batches += 1

        return {
            "train_loss": total_loss / max(1, num_batches),
            "l1_loss": total_l1 / max(1, num_batches),
            "stft_loss": total_stft / max(1, num_batches),
        }

    def fit(self, dataloader: DataLoader, epochs: int = 6) -> dict[str, list[float]]:
        """Trains the JEPA Separation Head across multiple epochs."""
        logger.info(f"Starting JEPA Separation Head training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            stats = self.train_epoch(dataloader)
            elapsed = time.time() - t0

            for k in self.history:
                self.history[k].append(stats[k])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Loss: {stats['train_loss']:.4f} (L1: {stats['l1_loss']:.4f}, STFT: {stats['stft_loss']:.4f})"
            )

            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "latest_jepa_sep.pt"
                self.save_checkpoint(ckpt_path, epoch=epoch)

        total_dur = time.time() - start_time
        logger.info(f"JEPA Separation Head training finished in {total_dur:.1f}s.")
        return self.history

    def save_checkpoint(self, path: str | Path, epoch: int | None = None) -> None:
        """Saves separation head weights, optimizer state, and training metadata."""
        ckpt = {
            "sep_head_state_dict": self.sep_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(ckpt, str(path))
        logger.debug(f"Saved JEPA separation checkpoint to: {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads separation head weights and history from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.sep_head.load_state_dict(ckpt["sep_head_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(f"Loaded JEPA separation checkpoint from: {path} (Epoch: {ckpt.get('epoch')})")
        return ckpt


__all__ = [
    "JEPASeparationTrainer",
]
