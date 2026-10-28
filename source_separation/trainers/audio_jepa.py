"""Audio-JEPA (Joint-Embedding Predictive Architecture) Trainer.

Provides self-supervised training for AudioJEPA via masked latent prediction and
exponential moving average (EMA) target encoder dynamics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseTrainer, logger


class AudioJEPATrainer(BaseTrainer):
    """Self-supervised trainer for AudioJEPA via masked latent predictive loss."""

    def __init__(
        self,
        jepa: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | None = None,
        checkpoint_dir: str | Path | None = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        resolved_device = device or self._resolve_device(jepa)
        super().__init__(device=resolved_device, checkpoint_dir=checkpoint_dir, max_grad_norm=max_grad_norm)

        self.jepa = jepa.to(self.device)
        self.optimizer = optimizer

        self.history: dict[str, list[float]] = {
            "loss": [],
            "cos_sim": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Runs one epoch of masked latent predictive pre-training."""
        self.jepa.train()
        total_loss = 0.0
        total_cos_sim = 0.0
        num_batches = 0

        for batch in dataloader:
            wav = (batch["target_audio"] if "target_audio" in batch else batch["target"]).to(self.device)
            out = self.jepa(wav)
            loss = out["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.jepa.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Update target encoder via EMA
            self.jepa.update_target_encoder()

            total_loss += float(loss.item())
            total_cos_sim += float(out["cos_sim"].item())
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        avg_cos_sim = total_cos_sim / max(1, num_batches)
        return {"loss": avg_loss, "cos_sim": avg_cos_sim}

    def fit(self, dataloader: DataLoader, epochs: int = 6) -> dict[str, list[float]]:
        """Trains the Audio-JEPA model across multiple epochs."""
        logger.info(f"Starting Audio-JEPA pretraining for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            stats = self.train_epoch(dataloader)
            elapsed = time.time() - t0

            self.history["loss"].append(stats["loss"])
            self.history["cos_sim"].append(stats["cos_sim"])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Predictive Loss: {stats['loss']:.4f}, Cosine Sim: {stats['cos_sim']:.4f}"
            )

            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "latest_jepa.pt"
                self.save_checkpoint(ckpt_path, epoch=epoch)

        total_dur = time.time() - start_time
        logger.info(f"Audio-JEPA pretraining finished in {total_dur:.1f}s.")
        return self.history

    def save_checkpoint(self, path: str | Path, epoch: int | None = None) -> None:
        """Saves JEPA weights, optimizer state, and pre-training trajectory."""
        ckpt = {
            "jepa_state_dict": self.jepa.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(ckpt, str(path))
        logger.debug(f"Saved Audio-JEPA checkpoint to: {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads JEPA weights and history from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.jepa.load_state_dict(ckpt["jepa_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(f"Loaded Audio-JEPA checkpoint from: {path} (Epoch: {ckpt.get('epoch')})")
        return ckpt


__all__ = [
    "AudioJEPATrainer",
]
