"""Energy-Based Model (EBM) Training Objectives and LatentEBMTrainer.

Provides Contrastive Divergence (CD) training for scalar energy-based models operating
over clean audio latent representations (e.g., from pre-trained Autoencoders or Beta-VAEs)
using Langevin MCMC dynamics and replay buffers.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .base import BaseTrainer, logger


class LatentContrastiveDivergenceLoss(nn.Module):
    """Contrastive Divergence (CD) loss with L2 energy regularization for ScalarLatentEBM.

    Formulation:
        L_CD = E_{z+ ~ p_data}[E(z+)] - E_{z- ~ p_model}[E(z-)] + lambda_reg * (E(z+)^2 + E(z-)^2)
    Minimizing L_CD decreases the energy of authentic audio latents (E(z+) -> low)
    and increases the energy of negative/noise samples (E(z-) -> high).
    """

    def __init__(self, energy_reg: float = 0.05) -> None:
        super().__init__()
        self.energy_reg = energy_reg

    def forward(self, pos_energy: Tensor, neg_energy: Tensor) -> Dict[str, Tensor]:
        pos_mean = pos_energy.mean()
        neg_mean = neg_energy.mean()
        cd_loss = pos_mean - neg_mean
        reg_loss = self.energy_reg * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())
        total = cd_loss + reg_loss
        return {
            "total_loss": total,
            "cd_loss": cd_loss,
            "reg_loss": reg_loss,
            "pos_energy": pos_mean,
            "neg_energy": neg_mean,
            "energy_gap": neg_mean - pos_mean,
        }


class LatentEBMTrainer(BaseTrainer):
    """Trainer for ScalarLatentEBM on clean instrument latents via Contrastive Divergence.

    Takes a pre-trained frozen audio encoder (e.g. from WaveformAutoencoder or WaveformBetaVAE),
    draws negative samples using short-run Langevin MCMC + persistent replay buffer, and
    optimizes the scalar energy landscape.
    """

    def __init__(
        self,
        ebm: nn.Module,
        encoder: nn.Module,
        replay_buffer: Any,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[LatentContrastiveDivergenceLoss] = None,
        device: Optional[torch.device] = None,
        mcmc_steps: int = 20,
        mcmc_step_size: float = 0.02,
        mcmc_noise_scale: float = 0.005,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        resolved_device = device or self._resolve_device(ebm)
        super().__init__(device=resolved_device, checkpoint_dir=checkpoint_dir, max_grad_norm=max_grad_norm)

        self.ebm = ebm.to(self.device)
        self.encoder = encoder
        if isinstance(self.encoder, nn.Module):
            self.encoder.to(self.device)
            self.encoder.eval()

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.criterion = criterion or LatentContrastiveDivergenceLoss()
        self.mcmc_steps = mcmc_steps
        self.mcmc_step_size = mcmc_step_size
        self.mcmc_noise_scale = mcmc_noise_scale

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "cd_loss": [],
            "reg_loss": [],
            "pos_energy": [],
            "neg_energy": [],
            "energy_gap": [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs one epoch of Contrastive Divergence training on latent audio representations."""
        self.ebm.train()
        epoch_stats = {
            "train_loss": 0.0,
            "cd_loss": 0.0,
            "reg_loss": 0.0,
            "pos_energy": 0.0,
            "neg_energy": 0.0,
            "energy_gap": 0.0,
        }
        num_batches = 0

        for batch in dataloader:
            # Extract clean target waveform
            target_wav = (batch["target_audio"] if "target_audio" in batch else batch["target"]).to(self.device)
            batch_size = target_wav.shape[0]

            # 1. Encode positive samples into latent space
            with torch.no_grad():
                if hasattr(self.encoder, "encode"):
                    pos_z = self.encoder.encode(target_wav)
                    if isinstance(pos_z, tuple):
                        pos_z = pos_z[0]  # Take mean mu if VAE
                else:
                    pos_z = self.encoder(target_wav)

            # 2. Sample negative candidates from replay buffer
            latent_shape = tuple(pos_z.shape[1:])
            neg_z_init, buffer_indices = self.replay_buffer.sample(
                batch_size=batch_size,
                latent_shape=latent_shape,
                device=self.device,
            )

            # 3. Short-run Langevin MCMC dynamics
            self.ebm.eval()
            neg_z = neg_z_init.clone().detach().requires_grad_(True)
            for _ in range(self.mcmc_steps):
                e = self.ebm(neg_z)
                grad = torch.autograd.grad(
                    outputs=e.sum(),
                    inputs=neg_z,
                    create_graph=False,
                    retain_graph=False,
                )[0]
                if self.max_grad_norm > 0:
                    grad = torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm)
                noise = torch.randn_like(neg_z) if self.mcmc_noise_scale > 0 else 0.0
                neg_z.data = (
                    neg_z.data
                    - 0.5 * self.mcmc_step_size * grad
                    + math.sqrt(self.mcmc_step_size) * self.mcmc_noise_scale * noise
                )

            neg_z = neg_z.detach()
            # Update replay buffer
            self.replay_buffer.update(neg_z, buffer_indices)

            # 4. Contrastive Divergence Loss
            self.ebm.train()
            pos_energy = self.ebm(pos_z)
            neg_energy = self.ebm(neg_z)

            loss_dict = self.criterion(pos_energy, neg_energy)
            total_loss = loss_dict["total_loss"]

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.ebm.parameters(), self.max_grad_norm)
            self.optimizer.step()

            epoch_stats["train_loss"] += float(loss_dict["total_loss"].item())
            epoch_stats["cd_loss"] += float(loss_dict["cd_loss"].item())
            epoch_stats["reg_loss"] += float(loss_dict["reg_loss"].item())
            epoch_stats["pos_energy"] += float(loss_dict["pos_energy"].item())
            epoch_stats["neg_energy"] += float(loss_dict["neg_energy"].item())
            epoch_stats["energy_gap"] += float(loss_dict["energy_gap"].item())
            num_batches += 1

        avg_stats = {k: v / max(1, num_batches) for k, v in epoch_stats.items()}
        return avg_stats

    def fit(self, train_loader: DataLoader, epochs: int = 10) -> Dict[str, List[float]]:
        """Trains the Latent EBM across multiple epochs."""
        logger.info(f"Starting EBM training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            stats = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            for k in self.history:
                self.history[k].append(stats[k])

            logger.info(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
                f"Loss: {stats['train_loss']:.4f} "
                f"(Pos E: {stats['pos_energy']:.3f}, Neg E: {stats['neg_energy']:.3f}, "
                f"Gap: {stats['energy_gap']:+.3f})"
            )

            if self.checkpoint_dir:
                ckpt_path = self.checkpoint_dir / "latest_ebm.pt"
                self.save_checkpoint(ckpt_path, epoch=epoch)

        total_dur = time.time() - start_time
        logger.info(f"EBM training finished in {total_dur:.1f}s.")
        return self.history

    def save_checkpoint(self, path: Union[str, Path], epoch: Optional[int] = None) -> None:
        """Saves EBM weights, optimizer state, and training trajectory."""
        ckpt = {
            "ebm_state_dict": self.ebm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "history": self.history,
        }
        torch.save(ckpt, str(path))
        logger.debug(f"Saved EBM checkpoint to: {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Loads EBM weights and history from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.ebm.load_state_dict(ckpt["ebm_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        logger.info(f"Loaded EBM checkpoint from: {path} (Epoch: {ckpt.get('epoch')})")
        return ckpt


__all__ = [
    "LatentContrastiveDivergenceLoss",
    "LatentEBMTrainer",
]
