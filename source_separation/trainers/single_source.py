"""Universal Architecture-Agnostic Single Source Separation Trainer and Objectives.

Provides a reusable training pipeline designed to train diverse single-source
separation model architectures (Autoencoders, VAEs, Beta-VAEs, U-Nets, Flows/Glow, Diffusion):

Features:
  - Architecture-Agnostic: Works with any model implementing forward(mix) -> recon.
  - Multi-Objective Loss: L1 Waveform Loss + Multi-Resolution STFT Spectral Loss.
  - Periodic Validation: Evaluates SNR, SI-SDR, and SI-SDRi on unseen evaluation corpora.
  - Checkpoint Management: Automatic best-model saving based on validation SI-SDR.
  - History Tracking: Records complete training trajectories for post-training visualization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .base import BaseTrainer, extract_reconstruction, logger

try:
    from ..evaluation import (
        MultiResolutionSTFTLoss,
        validate_model_during_training,
    )
except (ImportError, ValueError):
    try:
        from source_separation.evaluation import (
            MultiResolutionSTFTLoss,
            validate_model_during_training,
        )
    except ImportError:
        from evaluation import (
            MultiResolutionSTFTLoss,
            validate_model_during_training,
        )


class SingleSourceSeparationLoss(nn.Module):
    """Composite loss combining L1 time-domain waveform error and Multi-Resolution STFT spectral convergence."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(self, pred: Tensor | dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        if isinstance(pred, dict):
            pred = pred["recon"]
        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        total = self.alpha_l1 * l1 + self.alpha_stft * stft
        return {
            "total_loss": total,
            "l1_loss": l1,
            "stft_loss": stft,
        }


class SingleSourceVAELoss(nn.Module):
    """Composite loss for Single Source Separation VAEs: L1 + Multi-Resolution STFT + Analytical KL Divergence."""

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
        beta_kl: float = 1e-4,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.beta_kl = beta_kl
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_or_out: Tensor | dict[str, Tensor],
        target: Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(pred_or_out, dict):
            pred = pred_or_out["recon"]
            mu = pred_or_out.get("mu", None)
            logvar = pred_or_out.get("logvar", None)
        else:
            pred = pred_or_out
            mu = None
            logvar = None

        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        recon_loss = self.alpha_l1 * l1 + self.alpha_stft * stft

        if mu is not None and logvar is not None:
            # Analytical KL divergence: D_KL(q(z|x) || N(0, I))
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl = torch.tensor(0.0, device=pred.device)

        total = recon_loss + self.beta_kl * kl

        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "l1_loss": l1,
            "stft_loss": stft,
            "kl_loss": kl,
            "weighted_kl": self.beta_kl * kl,
        }


class SingleSourceBetaVAELoss(nn.Module):
    """Composite loss for Single Source Separation Beta-VAE (Higgins et al., 2017; Burgess et al., 2018; Lu et al., 2022).

    Formulation:
        L = alpha_l1 * L1 + alpha_stft * MR-STFT + beta * |D_KL - C|
    where:
        beta: Disentanglement pressure hyperparameter (beta >= 1.0).
        capacity (C): Controlled bottleneck capacity parameter. Gradually increasing C
        during training allows the network to learn progressively finer acoustic factors.
    """

    def __init__(
        self,
        alpha_l1: float = 1.0,
        alpha_stft: float = 0.5,
        beta: float = 4.0,
        beta_scale: float = 1e-4,
        capacity: float = 0.0,
    ) -> None:
        super().__init__()
        self.alpha_l1 = alpha_l1
        self.alpha_stft = alpha_stft
        self.beta = beta
        self.beta_scale = beta_scale
        self.capacity = capacity
        self.mr_stft = MultiResolutionSTFTLoss()

    def forward(
        self,
        pred_or_out: Tensor | dict[str, Tensor],
        target: Tensor,
    ) -> dict[str, Tensor]:
        if isinstance(pred_or_out, dict):
            pred = pred_or_out["recon"]
            mu = pred_or_out.get("mu", None)
            logvar = pred_or_out.get("logvar", None)
        else:
            pred = pred_or_out
            mu = None
            logvar = None

        l1 = F.l1_loss(pred, target)
        stft = self.mr_stft(pred, target)
        recon_loss = self.alpha_l1 * l1 + self.alpha_stft * stft

        if mu is not None and logvar is not None:
            # Analytical KL divergence
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        else:
            kl = torch.tensor(0.0, device=pred.device)

        eff_beta = self.beta * self.beta_scale
        if self.capacity > 0:
            kl_term = eff_beta * torch.abs(kl - self.capacity)
        else:
            kl_term = eff_beta * kl

        total = recon_loss + kl_term

        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "l1_loss": l1,
            "stft_loss": stft,
            "kl_loss": kl,
            "weighted_kl": kl_term,
            "eff_beta": torch.tensor(eff_beta, device=pred.device),
            "capacity": torch.tensor(self.capacity, device=pred.device),
        }


class SingleSourceTrainer(BaseTrainer):
    """Reusable training harness for Single Source Separation models.

    Allows training different model architectures (Deterministic Autoencoders,
    VAEs, Beta-VAEs, U-Nets, Normalizing Flows) using the exact same robust training,
    gradient clipping, validation, and checkpointing logic.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: nn.Module | None = None,
        scheduler: Any | None = None,
        grad_clip: float = 1.0,
        checkpoint_dir: str | Path | None = None,
        use_amp: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            checkpoint_dir=checkpoint_dir,
            max_grad_norm=grad_clip,
            use_amp=use_amp,
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or SingleSourceSeparationLoss()
        self.scheduler = scheduler
        self.grad_clip = grad_clip

        self.history: dict[str, list[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_l1": [],
            "train_stft": [],
            "train_kl": [],
            "val_loss": [],
            "val_snr": [],
            "val_si_sdr": [],
            "val_si_sdri": [],
        }
        self.best_si_sdr = -float("inf")
        self.best_model_path: Path | None = None

    @property
    def criterion(self) -> nn.Module:
        """Alias for loss_fn maintaining backward compatibility with evaluation runners."""
        return self.loss_fn

    @criterion.setter
    def criterion(self, value: nn.Module) -> None:
        self.loss_fn = value

    def _extract_reconstruction(self, model_output: Any) -> Tensor:
        """Standardizes model outputs from dict, tuple, or direct tensor."""
        return extract_reconstruction(model_output)

    def train_epoch(
        self,
        train_loader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Runs one full training epoch over the dataloader."""
        self.model.train()
        total_loss_sum = 0.0
        l1_sum = 0.0
        stft_sum = 0.0
        kl_sum = 0.0
        batches = 0

        for i, batch in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break

            mix = batch["mix_audio"].to(self.device)
            target = batch["target_audio"].to(self.device)

            self.optimizer.zero_grad()
            with self.autocast_context:
                out = self.model(mix)
                pred = self._extract_reconstruction(out)

                # Support loss functions taking full model dict (e.g. VAE with mu/logvar)
                # or functions taking just predicted reconstruction tensor
                try:
                    loss_dict = self.loss_fn(out, target)
                except TypeError:
                    loss_dict = self.loss_fn(pred, target)

                total_loss = loss_dict["total_loss"]

            if self.scaler.is_enabled():
                self.scaler.scale(total_loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            total_loss_sum += float(total_loss.item())
            l1_sum += float(loss_dict["l1_loss"].item())
            stft_sum += float(loss_dict["stft_loss"].item())
            if "kl_loss" in loss_dict:
                kl_sum += float(loss_dict["kl_loss"].item())
            batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        n = max(1, batches)
        return {
            "train_loss": total_loss_sum / n,
            "train_l1": l1_sum / n,
            "train_stft": stft_sum / n,
            "train_kl": kl_sum / n,
        }

    def evaluate(
        self,
        val_loader: DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, Any]:
        """Evaluates model performance on validation / cross-dataset loader."""
        return validate_model_during_training(
            model=self.model,
            val_loader=val_loader,
            criterion=self.criterion,
            device=self.device,
            max_batches=max_batches,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        val_every: int = 1,
        max_train_batches: int | None = None,
        max_val_batches: int | None = None,
        save_checkpoints: bool = True,
    ) -> dict[str, list[float]]:
        """Trains the model across multiple epochs with periodic validation."""
        logger.info(f"Starting training for {epochs} epochs on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(train_loader, max_batches=max_train_batches)

            val_metrics: dict[str, Any] | None = None
            if epoch % val_every == 0 or epoch == epochs:
                val_metrics = self.evaluate(val_loader, max_batches=max_val_batches)

            ep_duration = time.time() - ep_start

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_l1"].append(train_metrics["train_l1"])
            self.history["train_stft"].append(train_metrics["train_stft"])
            self.history["train_kl"].append(train_metrics.get("train_kl", 0.0))

            log_msg = (
                f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] - "
                f"Train Loss: {train_metrics['train_loss']:.4f} (L1: {train_metrics['train_l1']:.4f}, STFT: {train_metrics['train_stft']:.4f}"
            )
            if train_metrics.get("train_kl", 0.0) > 0:
                log_msg += f", KL: {train_metrics['train_kl']:.4f}"
            log_msg += ")"

            if val_metrics is not None:
                val_snr = val_metrics["snr_mean"]
                val_si_sdr = val_metrics["si_sdr_mean"]
                val_si_sdri = val_metrics["si_sdri_mean"]
                val_l1 = val_metrics["l1_mean"]

                self.history["val_loss"].append(val_l1)
                self.history["val_snr"].append(val_snr)
                self.history["val_si_sdr"].append(val_si_sdr)
                self.history["val_si_sdri"].append(val_si_sdri)

                log_msg += f" | Val SI-SDR: {val_si_sdr:+.2f} dB, SNR: {val_snr:+.2f} dB, SI-SDRi: {val_si_sdri:+.2f} dB"

                # Check for best model
                if val_si_sdr > self.best_si_sdr:
                    self.best_si_sdr = val_si_sdr
                    if save_checkpoints and self.checkpoint_dir:
                        self.best_model_path = self.checkpoint_dir / "best_model.pt"
                        self.save_checkpoint(self.best_model_path, epoch=epoch, si_sdr=val_si_sdr)
                        log_msg += f" -> [Saved Best Model: SI-SDR = {val_si_sdr:+.2f} dB]"

            logger.info(log_msg)

        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration:.1f} seconds. Best Val SI-SDR: {self.best_si_sdr:+.2f} dB")
        return self.history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int | None = None,
        si_sdr: float | None = None,
    ) -> None:
        """Saves model weights, optimizer state, and training metadata."""
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_si_sdr": si_sdr if si_sdr is not None else self.best_si_sdr,
            "history": self.history,
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler.is_enabled():
            ckpt["scaler_state_dict"] = self.scaler.state_dict()
        torch.save(ckpt, str(path))
        logger.debug(f"Saved checkpoint to: {path}")

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads model weights and optimizer state from checkpoint."""
        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler is not None and hasattr(self.scheduler, "load_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        if "best_si_sdr" in ckpt:
            self.best_si_sdr = ckpt["best_si_sdr"]
        logger.info(f"Loaded checkpoint from: {path} (Epoch: {ckpt.get('epoch')}, SI-SDR: {ckpt.get('best_si_sdr'):+.2f} dB)")
        return ckpt


__all__ = [
    "SingleSourceBetaVAELoss",
    "SingleSourceSeparationLoss",
    "SingleSourceTrainer",
    "SingleSourceVAELoss",
]
