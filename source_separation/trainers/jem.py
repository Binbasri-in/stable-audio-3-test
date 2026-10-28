"""Joint Energy Model (JEM) Training Objectives and WaveformJEMTrainer.

Provides joint discriminative and generative training for WaveformJEM on multi-instrument
single sources via Joint Energy Learning (classification + free-energy contrastive divergence
with short-run SGLD MCMC dynamics).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .base import BaseTrainer, logger


class WaveformJEMLoss(nn.Module):
    """Joint Energy Model (JEM) objective balancing classification and free-energy CD.

    Formulation:
        L_JEM = L_classification + alpha_energy * L_energy
    where:
        L_classification = CrossEntropy(logits, target_class)
        L_energy = E(x+) - E(x-) + lambda_reg * (E(x+)^2 + E(x-)^2)
    """

    def __init__(self, alpha_energy: float = 0.2, energy_reg: float = 0.05) -> None:
        super().__init__()
        self.alpha_energy = alpha_energy
        self.energy_reg = energy_reg
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        pos_energy: Tensor,
        neg_energy: Tensor,
    ) -> dict[str, Tensor]:
        clf_loss = self.ce_loss(logits, targets)
        pos_mean = pos_energy.mean()
        neg_mean = neg_energy.mean()
        cd_loss = pos_mean - neg_mean
        reg_loss = self.energy_reg * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())
        energy_loss = cd_loss + reg_loss
        total = clf_loss + self.alpha_energy * energy_loss

        return {
            "total_loss": total,
            "clf_loss": clf_loss,
            "energy_loss": energy_loss,
            "cd_loss": cd_loss,
            "reg_loss": reg_loss,
            "pos_energy": pos_mean,
            "neg_energy": neg_mean,
            "energy_gap": neg_mean - pos_mean,
        }


class WaveformJEMTrainer(BaseTrainer):
    """Trainer for WaveformJEM on multi-instrument single sources via Joint Energy Learning."""

    def __init__(
        self,
        jem: nn.Module,
        replay_buffer: Any,
        optimizer: torch.optim.Optimizer,
        criterion: WaveformJEMLoss | None = None,
        device: torch.device | None = None,
        mcmc_steps: int = 10,
        mcmc_step_size: float = 0.02,
        mcmc_noise_scale: float = 0.005,
        checkpoint_dir: str | Path | None = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        resolved_device = device or self._resolve_device(jem)
        super().__init__(device=resolved_device, checkpoint_dir=checkpoint_dir, max_grad_norm=max_grad_norm)

        self.jem = jem.to(self.device)
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.criterion = criterion or WaveformJEMLoss()
        self.mcmc_steps = mcmc_steps
        self.mcmc_step_size = mcmc_step_size
        self.mcmc_noise_scale = mcmc_noise_scale

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "clf_loss": [],
            "energy_loss": [],
            "pos_energy": [],
            "neg_energy": [],
            "energy_gap": [],
            "accuracy": [],
        }

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        class_mapping: dict[str, int],
    ) -> dict[str, float]:
        """Runs one epoch of Joint Energy Model training across multi-stem dataloaders."""
        self.jem.train()
        epoch_stats = {
            "train_loss": 0.0,
            "clf_loss": 0.0,
            "energy_loss": 0.0,
            "pos_energy": 0.0,
            "neg_energy": 0.0,
            "energy_gap": 0.0,
            "accuracy": 0.0,
        }
        num_batches = 0

        # Create combined iterator over available instrument dataloaders
        iterators = {k: iter(dl) for k, dl in dataloaders.items()}
        max_batches = max(len(dl) for dl in dataloaders.values())

        for _ in range(max_batches):
            batch_audios = []
            batch_labels = []

            for name, it in iterators.items():
                try:
                    b = next(it)
                except StopIteration:
                    iterators[name] = iter(dataloaders[name])
                    b = next(iterators[name])

                wav = (b["target_audio"] if "target_audio" in b else b["target"]).to(self.device)
                cls_idx = class_mapping[name]
                batch_audios.append(wav)
                batch_labels.append(torch.full((wav.size(0),), cls_idx, dtype=torch.long, device=self.device))

            x_pos = torch.cat(batch_audios, dim=0)
            y_pos = torch.cat(batch_labels, dim=0)
            b_size = x_pos.shape[0]

            # 1. Sample negative waveform candidates from replay buffer
            audio_shape = tuple(x_pos.shape[1:])
            x_neg_init, buffer_indices = self.replay_buffer.sample(
                batch_size=b_size,
                audio_shape=audio_shape,
                device=self.device,
            )

            # 2. Short-run SGLD dynamics on free energy E_phi(x)
            self.jem.eval()
            x_neg = x_neg_init.clone().detach().requires_grad_(True)
            for _ in range(self.mcmc_steps):
                energy = self.jem.compute_free_energy(x_neg)
                grad = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=x_neg,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                if self.max_grad_norm > 0:
                    grad = torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm)
                noise = torch.randn_like(x_neg) if self.mcmc_noise_scale > 0 else 0.0
                x_neg.data = (
                    x_neg.data
                    - 0.5 * self.mcmc_step_size * grad
                    + math.sqrt(self.mcmc_step_size) * self.mcmc_noise_scale * noise
                )

            x_neg = x_neg.detach()
            self.replay_buffer.update(x_neg, buffer_indices)

            # 3. Forward Pass & Loss
            self.jem.train()
            logits_pos = self.jem(x_pos)
            pos_energy = self.jem.compute_free_energy(x_pos)
            neg_energy = self.jem.compute_free_energy(x_neg)

            loss_dict = self.criterion(logits_pos, y_pos, pos_energy, neg_energy)
            total_loss = loss_dict["total_loss"]

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.jem.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Compute classification accuracy
            preds = logits_pos.argmax(dim=-1)
            acc = (preds == y_pos).float().mean()

            epoch_stats["train_loss"] += float(loss_dict["total_loss"].item())
            epoch_stats["clf_loss"] += float(loss_dict["clf_loss"].item())
            epoch_stats["energy_loss"] += float(loss_dict["energy_loss"].item())
            epoch_stats["pos_energy"] += float(loss_dict["pos_energy"].item())
            epoch_stats["neg_energy"] += float(loss_dict["neg_energy"].item())
            epoch_stats["energy_gap"] += float(loss_dict["energy_gap"].item())
            epoch_stats["accuracy"] += float(acc.item())
            num_batches += 1

        avg_stats = {k: v / max(1, num_batches) for k, v in epoch_stats.items()}
        return avg_stats

    def fit(
        self,
        dataloaders: dict[str, DataLoader],
        class_mapping: dict[str, int],
        epochs: int = 6,
    ) -> dict[str, list[float]]:
        """Trains the Waveform JEM across multiple epochs."""
        logger.info(f"Starting JEM training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            stats = self.train_epoch(dataloaders, class_mapping)
            elapsed = time.time() - t0

            for k in self.history:
                self.history[k].append(stats[k])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Loss: {stats['train_loss']:.4f} "
                f"(Clf: {stats['clf_loss']:.3f}, Acc: {stats['accuracy']*100:.1f}%, "
                f"Pos E: {stats['pos_energy']:.3f}, Neg E: {stats['neg_energy']:.3f}, "
                f"Gap: {stats['energy_gap']:+.3f})"
            )

            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "latest_jem.pt"
                self.save_checkpoint(ckpt_path, epoch=epoch)

        total_dur = time.time() - start_time
        logger.info(f"JEM training finished in {total_dur:.1f}s.")
        return self.history

    def save_checkpoint(self, path: str | Path, epoch: int | None = None) -> None:
        """Saves JEM weights, optimizer state, and training metadata."""
        ckpt = {
            "jem_state_dict": self.jem.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(ckpt, str(path))
        logger.debug(f"Saved JEM checkpoint to: {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads JEM weights and history from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.jem.load_state_dict(ckpt["jem_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(f"Loaded JEM checkpoint from: {path} (Epoch: {ckpt.get('epoch')})")
        return ckpt


__all__ = [
    "WaveformJEMLoss",
    "WaveformJEMTrainer",
]
