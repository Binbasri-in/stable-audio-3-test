"""Training Validation Suite for Audio Source Separation.

Provides dedicated, high-performance, GPU-accelerated validation methods
specifically designed to run periodically during training loops (per-epoch or per-step):

1. `validate_model_during_training`:
   - Complete batch-optimized validation runner across a validation DataLoader.
   - Computes multi-objective validation losses (L1 + MultiResolution STFT).
   - Evaluates standard time-domain metrics: SNR, SI-SDR, SI-SDR improvement (SI-SDRi).
   - Optional CDX utterance SDR calculation.
   - Extracts qualitative sample bundles (mixture, target, reconstructed stem, residual)
     for direct human playback, Colab audio widgets, TensorBoard, or WandB logging.
   - Drop-in compatible with `SingleSourceTrainer`.

2. `validate_batch`:
   - Single-batch inline validation routine for quick step-level debugging and validation hooks.

3. `SeparationTrainingValidator`:
   - Stateful validation and early-stopping manager.
   - Tracks best metrics, history trajectories, checkpoint triggers, and patience counters.
   - Generates pretty log strings and interactive Colab / Jupyter audio audit widgets.

4. `evaluate_source_separation`:
   - Standard evaluation runner maintaining 100% backward compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .cdx import compute_utterance_sdr
from .common import logger
from .metrics import (
    MultiResolutionSTFTLoss,
    compute_si_sdr,
    compute_si_sdr_improvement,
    compute_snr,
    display_audio_player,
)


@dataclass
class TrainingValidationResult:
    """Structured container holding metrics and audit samples from a validation run."""

    epoch: Optional[int]
    loss: float
    l1_loss: float
    stft_loss: float
    snr_mean: float
    snr_std: float
    si_sdr_mean: float
    si_sdr_std: float
    si_sdri_mean: float
    si_sdri_std: float
    cdx_sdr_mean: Optional[float] = None
    cdx_sdr_std: Optional[float] = None
    num_samples: int = 0
    duration_seconds: float = 0.0
    is_best: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    samples: Optional[Dict[str, Tensor]] = None

    def summary(self) -> str:
        """Returns a concise, publication-grade log string for terminal and log files."""
        ep_prefix = f"Epoch {self.epoch:02d} | " if self.epoch is not None else ""
        msg = (
            f"{ep_prefix}Val Loss: {self.loss:.4f} (L1: {self.l1_loss:.4f}, STFT: {self.stft_loss:.4f}) | "
            f"SI-SDR: {self.si_sdr_mean:+.2f} ± {self.si_sdr_std:.2f} dB | "
            f"SNR: {self.snr_mean:+.2f} dB | SI-SDRi: {self.si_sdri_mean:+.2f} dB"
        )
        if self.cdx_sdr_mean is not None:
            msg += f" | CDX SDR: {self.cdx_sdr_mean:+.2f} dB"
        if self.is_best:
            msg += " -> [★ New Best Model]"
        return msg


def _extract_model_output(output: Any) -> Tensor:
    """Helper to extract estimated waveform tensor from arbitrary model outputs."""
    if isinstance(output, dict):
        return output.get("recon", output.get("estimate", output.get("waveform", next(iter(output.values())))))
    elif isinstance(output, (list, tuple)):
        return output[0]
    elif isinstance(output, Tensor):
        return output
    raise ValueError(f"Unsupported model output type: {type(output)}")


def validate_batch(
    model: nn.Module,
    batch: Dict[str, Any],
    criterion: Optional[Union[nn.Module, Callable]] = None,
    device: Optional[Union[torch.device, str]] = None,
    compute_metrics: bool = True,
    mix_key: str = "mix_audio",
    target_key: str = "target_audio",
) -> Dict[str, Any]:
    """Validates model on a single batch during training.

    Args:
        model: Separation neural network.
        batch: Dictionary from DataLoader containing mixture and clean target audio.
        criterion: Optional loss function (takes pred, target).
        device: Device to run validation on.
        compute_metrics: Whether to compute SNR, SI-SDR, SI-SDRi.
        mix_key: Dictionary key for mixture waveform.
        target_key: Dictionary key for clean target waveform.

    Returns:
        Dictionary of scalar metrics and prediction tensors.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    elif isinstance(device, str):
        device = torch.device(device)

    with torch.no_grad():
        mix = batch[mix_key].to(device)
        target = batch[target_key].to(device)

        out = model(mix)
        estimate = _extract_model_output(out)

        # Standardize channel dimension
        if estimate.shape[-1] != target.shape[-1]:
            min_len = min(estimate.shape[-1], target.shape[-1])
            estimate = estimate[..., :min_len]
            target = target[..., :min_len]
            mix = mix[..., :min_len]

        l1 = F.l1_loss(estimate, target).item()
        res: Dict[str, Any] = {
            "l1_loss": l1,
            "loss": l1,
            "estimate": estimate,
            "target": target,
            "mixture": mix,
        }

        if criterion is not None:
            c_loss = criterion(out if isinstance(out, dict) else estimate, target)
            if isinstance(c_loss, dict):
                res["loss"] = c_loss["total_loss"].item()
                for k, v in c_loss.items():
                    res[k] = v.item() if isinstance(v, Tensor) else float(v)
            elif isinstance(c_loss, Tensor):
                res["loss"] = c_loss.item()

        if compute_metrics:
            snr = compute_snr(target, estimate)
            si_sdr = compute_si_sdr(target, estimate)
            si_sdri = compute_si_sdr_improvement(target, estimate, mix)

            res["snr"] = float(snr.mean().item())
            res["si_sdr"] = float(si_sdr.mean().item())
            res["si_sdri"] = float(si_sdri.mean().item())

    return res


def validate_model_during_training(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: Optional[Union[nn.Module, Callable]] = None,
    device: Optional[Union[torch.device, str]] = None,
    max_batches: Optional[int] = None,
    compute_si_sdri: bool = True,
    compute_cdx_sdr: bool = False,
    compute_stft_loss: bool = True,
    return_samples: bool = True,
    sample_count: int = 3,
    mix_key: str = "mix_audio",
    target_key: str = "target_audio",
    epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs a complete, fast validation cycle over a DataLoader during training.

    Designed for lightweight periodic execution in trainer loops (e.g. `SingleSourceTrainer`,
    custom PyTorch training loops, or Lightning/Accelerate).

    Features:
    - Zero gradient computation (`torch.no_grad()`).
    - Bounded runtime via `max_batches` (e.g., test 15-20 batches per epoch for fast validation).
    - Multi-metric tracking: L1 error, STFT spectral loss, SNR, SI-SDR, SI-SDR improvement.
    - Optional Utterance-level Global SDR (CDX Paper 2).
    - Qualitative sample caching: extracts audio sample tuples (mixture, target, prediction, error)
      for interactive playback or logging.
    - 100% backward compatibility: exports keys `snr_mean`, `si_sdr_mean`, `si_sdri_mean`, `l1_mean`.

    Args:
        model: Separation model to validate.
        val_loader: Validation DataLoader.
        criterion: Optional loss module or callable.
        device: Device to run evaluation on. If None, inferred from model parameters.
        max_batches: Maximum batches to process (None for entire DataLoader).
        compute_si_sdri: Whether to compute improvement over unprocessed mixture.
        compute_cdx_sdr: Whether to compute SDX'23 CDX utterance SDR.
        compute_stft_loss: Whether to compute MultiResolutionSTFTLoss if not in criterion.
        return_samples: Whether to store qualitative audio sample bundle.
        sample_count: Number of sample audio tracks to retain for human listening.
        mix_key: Batch dictionary key for mixture audio.
        target_key: Batch dictionary key for clean target audio.
        epoch: Optional current training epoch index.

    Returns:
        Structured dictionary of validation results.
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    elif isinstance(device, str):
        device = torch.device(device)

    start_time = time.time()

    mr_stft = MultiResolutionSTFTLoss() if compute_stft_loss else None

    loss_list: List[float] = []
    l1_list: List[float] = []
    stft_list: List[float] = []
    snr_list: List[float] = []
    si_sdr_list: List[float] = []
    si_sdri_list: List[float] = []
    cdx_sdr_list: List[float] = []

    sample_bundle: Optional[Dict[str, Tensor]] = None

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            mix = batch[mix_key].to(device)
            target = batch[target_key].to(device)

            out = model(mix)
            estimate = _extract_model_output(out)

            # Match lengths if slight discrepancy exists
            if estimate.shape[-1] != target.shape[-1]:
                min_len = min(estimate.shape[-1], target.shape[-1])
                estimate = estimate[..., :min_len]
                target = target[..., :min_len]
                mix = mix[..., :min_len]

            # 1. Losses
            batch_l1 = F.l1_loss(estimate, target).item()
            l1_list.append(batch_l1)

            batch_stft = 0.0
            if mr_stft is not None:
                try:
                    batch_stft = mr_stft(estimate, target).item()
                    stft_list.append(batch_stft)
                except Exception:
                    pass

            if criterion is not None:
                c_out = criterion(out if isinstance(out, dict) else estimate, target)
                if isinstance(c_out, dict):
                    loss_list.append(c_out["total_loss"].item())
                elif isinstance(c_out, Tensor):
                    loss_list.append(c_out.item())
                else:
                    loss_list.append(float(c_out))
            else:
                loss_list.append(batch_l1 + (0.5 * batch_stft if batch_stft > 0 else 0.0))

            # 2. Separation fidelity metrics
            snr = compute_snr(target, estimate)
            si_sdr = compute_si_sdr(target, estimate)

            snr_list.extend(snr.cpu().tolist())
            si_sdr_list.extend(si_sdr.cpu().tolist())

            if compute_si_sdri:
                si_sdri = compute_si_sdr_improvement(target, estimate, mix)
                si_sdri_list.extend(si_sdri.cpu().tolist())

            if compute_cdx_sdr:
                for b in range(target.shape[0]):
                    c_sdr = compute_utterance_sdr(target[b], estimate[b])
                    cdx_sdr_list.append(c_sdr)

            # 3. Cache sample bundle from first batch for human listening
            if return_samples and sample_bundle is None:
                k = min(sample_count, mix.shape[0])
                sample_bundle = {
                    "mixture": mix[:k].detach().cpu(),
                    "target": target[:k].detach().cpu(),
                    "estimate": estimate[:k].detach().cpu(),
                    "residual": (mix - estimate)[:k].detach().cpu(),
                    "error": (target - estimate)[:k].detach().cpu(),
                    "track_ids": batch.get("track_ids", ["Track"] * k)[:k],
                    "stem_names": batch.get("stem_names", ["stem"] * k)[:k],
                }

    elapsed = time.time() - start_time
    n_samples = len(snr_list)

    res: Dict[str, Any] = {
        # Primary validation losses
        "val_loss": float(np.mean(loss_list)) if loss_list else 0.0,
        "val_loss_std": float(np.std(loss_list)) if loss_list else 0.0,
        "l1_mean": float(np.mean(l1_list)) if l1_list else 0.0,
        "val_l1": float(np.mean(l1_list)) if l1_list else 0.0,
        "stft_mean": float(np.mean(stft_list)) if stft_list else 0.0,
        "val_stft": float(np.mean(stft_list)) if stft_list else 0.0,
        # Separation Metrics (mean and standard deviation)
        "snr_mean": float(np.mean(snr_list)) if snr_list else 0.0,
        "snr_std": float(np.std(snr_list)) if snr_list else 0.0,
        "val_snr": float(np.mean(snr_list)) if snr_list else 0.0,
        "si_sdr_mean": float(np.mean(si_sdr_list)) if si_sdr_list else 0.0,
        "si_sdr_std": float(np.std(si_sdr_list)) if si_sdr_list else 0.0,
        "val_si_sdr": float(np.mean(si_sdr_list)) if si_sdr_list else 0.0,
        "si_sdri_mean": float(np.mean(si_sdri_list)) if si_sdri_list else 0.0,
        "si_sdri_std": float(np.std(si_sdri_list)) if si_sdri_list else 0.0,
        "val_si_sdri": float(np.mean(si_sdri_list)) if si_sdri_list else 0.0,
        # Execution Metadata
        "num_evaluated_samples": n_samples,
        "num_evaluated_batches": min(len(val_loader), max_batches) if max_batches else len(val_loader),
        "duration_seconds": elapsed,
        "epoch": epoch,
        "samples": sample_bundle,
    }

    if compute_cdx_sdr and cdx_sdr_list:
        res["cdx_utterance_sdr_mean"] = float(np.mean(cdx_sdr_list))
        res["cdx_utterance_sdr_std"] = float(np.std(cdx_sdr_list))
        res["val_cdx_sdr"] = float(np.mean(cdx_sdr_list))

    return res


class SeparationTrainingValidator:
    """Stateful validation engine and early-stopping manager for source separation training.

    Tracks validation history trajectories, checks for performance improvement,
    automates best-model checkpointing, and provides audio audit tools for Jupyter/Colab.

    Example Usage:
    ```python
    validator = SeparationTrainingValidator(
        checkpoint_dir="./checkpoints",
        metric_to_track="si_sdr",
        mode="max",
        patience=10,
    )

    for epoch in range(1, epochs + 1):
        train_one_epoch(...)
        if validator.should_validate(epoch):
            result = validator.validate(model, val_loader, epoch=epoch, max_batches=20)
            logger.info(result.summary())
            if result.is_best:
                validator.save_best_checkpoint(model, optimizer, epoch=epoch)
            if validator.should_stop():
                logger.info("Early stopping triggered!")
                break
    ```
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        val_every: int = 1,
        metric_to_track: str = "si_sdr",
        mode: str = "max",
        patience: Optional[int] = None,
        min_delta: float = 0.01,
        criterion: Optional[Union[nn.Module, Callable]] = None,
        compute_cdx_sdr: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.val_every = max(1, val_every)
        self.metric_to_track = metric_to_track.lower()
        self.mode = mode.lower()
        self.patience = patience
        self.min_delta = min_delta
        self.criterion = criterion
        self.compute_cdx_sdr = compute_cdx_sdr

        # State tracking
        self.best_score: float = float("-inf") if self.mode == "max" else float("inf")
        self.best_epoch: Optional[int] = None
        self.patience_counter: int = 0
        self.last_result: Optional[TrainingValidationResult] = None

        # Trajectory history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "val_loss": [],
            "val_l1": [],
            "val_stft": [],
            "val_snr": [],
            "val_si_sdr": [],
            "val_si_sdri": [],
        }
        if self.compute_cdx_sdr:
            self.history["val_cdx_sdr"] = []

    def should_validate(self, epoch: int) -> bool:
        """Determines whether validation should execute at the current epoch."""
        return (epoch % self.val_every == 0)

    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        epoch: Optional[int] = None,
        max_batches: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> TrainingValidationResult:
        """Executes validation cycle and updates internal state and history."""
        raw_res = validate_model_during_training(
            model=model,
            val_loader=val_loader,
            criterion=self.criterion,
            device=device,
            max_batches=max_batches,
            compute_cdx_sdr=self.compute_cdx_sdr,
            epoch=epoch,
        )

        # Determine target score
        if "si_sdr" in self.metric_to_track:
            current_score = raw_res["si_sdr_mean"]
        elif "snr" in self.metric_to_track:
            current_score = raw_res["snr_mean"]
        elif "loss" in self.metric_to_track or "l1" in self.metric_to_track:
            current_score = raw_res["val_loss"]
        elif "cdx" in self.metric_to_track:
            current_score = raw_res.get("cdx_utterance_sdr_mean", 0.0)
        else:
            current_score = raw_res["si_sdr_mean"]

        # Check for improvement
        is_best = False
        if self.mode == "max":
            if current_score > (self.best_score + self.min_delta):
                self.best_score = current_score
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
        else:
            if current_score < (self.best_score - self.min_delta):
                self.best_score = current_score
                self.best_epoch = epoch
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1

        # Record history
        if epoch is not None:
            self.history["epoch"].append(epoch)
        self.history["val_loss"].append(raw_res["val_loss"])
        self.history["val_l1"].append(raw_res["l1_mean"])
        self.history["val_stft"].append(raw_res["stft_mean"])
        self.history["val_snr"].append(raw_res["snr_mean"])
        self.history["val_si_sdr"].append(raw_res["si_sdr_mean"])
        self.history["val_si_sdri"].append(raw_res["si_sdri_mean"])
        if self.compute_cdx_sdr:
            self.history["val_cdx_sdr"].append(raw_res.get("cdx_utterance_sdr_mean", 0.0))

        result = TrainingValidationResult(
            epoch=epoch,
            loss=raw_res["val_loss"],
            l1_loss=raw_res["l1_mean"],
            stft_loss=raw_res["stft_mean"],
            snr_mean=raw_res["snr_mean"],
            snr_std=raw_res["snr_std"],
            si_sdr_mean=raw_res["si_sdr_mean"],
            si_sdr_std=raw_res["si_sdr_std"],
            si_sdri_mean=raw_res["si_sdri_mean"],
            si_sdri_std=raw_res["si_sdri_std"],
            cdx_sdr_mean=raw_res.get("cdx_utterance_sdr_mean"),
            cdx_sdr_std=raw_res.get("cdx_utterance_sdr_std"),
            num_samples=raw_res["num_evaluated_samples"],
            duration_seconds=raw_res["duration_seconds"],
            is_best=is_best,
            metrics=raw_res,
            samples=raw_res.get("samples"),
        )
        self.last_result = result
        return result

    def should_stop(self) -> bool:
        """Checks if early stopping patience threshold has been reached."""
        if self.patience is None:
            return False
        return self.patience_counter >= self.patience

    def save_best_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        path: Optional[Union[str, Path]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Saves weights and metadata of the best performing checkpoint."""
        if path is None:
            if self.checkpoint_dir is None:
                raise ValueError("Must provide either `path` or initialize with `checkpoint_dir`.")
            save_path = self.checkpoint_dir / "best_model.pt"
        else:
            save_path = Path(path)

        state = {
            "epoch": epoch,
            "best_score": self.best_score,
            "metric_tracked": self.metric_to_track,
            "model_state_dict": model.state_dict(),
            "history": self.history,
        }
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if extra_metadata is not None:
            state["metadata"] = extra_metadata

        torch.save(state, save_path)
        logger.info(f"Saved best model checkpoint to: {save_path} (Score: {self.best_score:+.2f})")
        return save_path

    def display_sample_audit(self, sample_idx: int = 0, sample_rate: int = 44100) -> None:
        """Renders playable audio widgets for the latest validation sample predictions."""
        if self.last_result is None or self.last_result.samples is None:
            print("No validation samples available yet.")
            return

        samples = self.last_result.samples
        k = min(sample_idx, len(samples["mixture"]) - 1)

        track_id = samples["track_ids"][k] if "track_ids" in samples else f"Track #{k}"
        stem = samples["stem_names"][k] if "stem_names" in samples else "Target"

        print("=" * 65)
        print(f"  Validation Listener Audit: {track_id} (Stem: {stem})")
        print(f"  Val SI-SDR: {self.last_result.si_sdr_mean:+.2f} dB | SNR: {self.last_result.snr_mean:+.2f} dB")
        print("=" * 65)

        display_audio_player(samples["mixture"][k], sample_rate, f"1. Mixture ({track_id})")
        display_audio_player(samples["target"][k], sample_rate, f"2. Ground Truth ({stem})")
        display_audio_player(samples["estimate"][k], sample_rate, f"3. Model Separated ({stem})")
        display_audio_player(samples["error"][k], sample_rate, f"4. Error (Target - Estimate)")


# ===========================================================================
# 5. General High-Level Evaluation Runner (Preserved for Trainer Compatibility)
# ===========================================================================


def evaluate_source_separation(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    max_batches: Optional[int] = None,
    compute_cdx_sdr: bool = False,
) -> Dict[str, Any]:
    """Evaluates source separation model across a dataset, returning metrics and sample tensors.

    Computes:
      - Mean SNR (dB) & Standard Deviation
      - Mean SI-SDR (dB) & Standard Deviation
      - Mean SI-SDR Improvement (dB) over raw mixture
      - Mean L1 Waveform Error
      - Optional Utterance-level Global SDR (CDX Paper 2)
      - First batch sample predictions for human listening
    """
    return validate_model_during_training(
        model=model,
        val_loader=dataloader,
        criterion=None,
        device=device,
        max_batches=max_batches,
        compute_si_sdri=True,
        compute_cdx_sdr=compute_cdx_sdr,
        return_samples=True,
    )
