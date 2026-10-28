"""Joint Energy Models (JEM) for Single-Source Separation & Prior Extraction.

Implements unified discriminative-generative Joint Energy Models (Grathwohl et al., 2020)
for waveform single-source separation, cross-stem bleed suppression, and energy-guided
controllable audio generation:
  1. JEMResidualBlock1D: Dilated residual 1D convolution block with SiLU activations.
  2. WaveformJEM: Joint classifier and energy-based model over raw waveforms.
  3. JEMReplayBuffer: Persistent MCMC replay buffer for SGLD sampling.
  4. JEMSingleSourceSeparator: Test-time single-source demixer with bleed suppression barrier.
  5. JEMGenerativeSteeringBridge: Bridges JEM acoustic priors to diffusion models (e.g. AdaLN injection).
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from ..evaluation import MultiResolutionSTFTLoss
except (ImportError, ValueError):
    try:
        from source_separation.evaluation import MultiResolutionSTFTLoss
    except ImportError:
        from evaluation import MultiResolutionSTFTLoss


# ---------------------------------------------------------------------------
# 1. Smooth 1D Residual Blocks for Waveform JEM
# ---------------------------------------------------------------------------


class JEMResidualBlock1D(nn.Module):
    """Dilated residual 1D convolution block with SiLU activations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=1,
            padding=kernel_size // 2,
            bias=True,
        )
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + residual)


# ---------------------------------------------------------------------------
# 2. Waveform Joint Energy Model (WaveformJEM)
# ---------------------------------------------------------------------------


class WaveformJEM(nn.Module):
    """Unified Waveform Classifier and Energy-Based Model.

    Evaluates:
      1. Class logits: f_phi(x) in R^{B, num_classes}
      2. Free energy:  E_phi(x) = -LogSumExp(f_phi(x))
      3. Class energy: E_phi(x | c) = -f_phi(x)[c]
      4. Score vector: nab_x log p(c | x) for generative diffusion control.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_classes: int = 3,
        num_stages: int = 4,
        strides: tuple[int, ...] = (4, 4, 4, 4),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # 1. Downsampling front-end with strided convolutions
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels
        ch = base_channels

        for i, stride in enumerate(strides[:num_stages]):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, ch, kernel_size=stride * 2, stride=stride, padding=stride // 2),
                    nn.SiLU(),
                    JEMResidualBlock1D(ch, kernel_size=5, dilation=2**i),
                )
            )
            in_ch = ch
            ch = min(256, ch * 2)

        self.final_channels = in_ch

        # 2. Global temporal aggregation and classification head
        self.conv_out = nn.Conv1d(self.final_channels, self.final_channels, kernel_size=1)
        self.act_out = nn.SiLU()
        self.classifier_head = nn.Linear(self.final_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Computes instrument class logits f_phi(x) in R^{B, num_classes}."""
        if x.ndim == 2:
            x = x.unsqueeze(1)

        h = x
        for block in self.down_blocks:
            h = block(h)

        h = self.act_out(self.conv_out(h))

        h_avg = h.mean(dim=-1)
        h_max = h.amax(dim=-1)
        h_pool = 0.5 * (h_avg + h_max)

        logits = self.classifier_head(h_pool)
        return logits

    def compute_free_energy(self, x: Tensor) -> Tensor:
        """Computes unnormalized marginal free energy:
            E_phi(x) = -LogSumExp_{c} (f_phi(x)[c])
        """
        logits = self.forward(x)
        free_energy = -torch.logsumexp(logits, dim=-1, keepdim=True)
        return free_energy

    def compute_class_energy(self, x: Tensor, class_idx: int) -> Tensor:
        """Computes class-conditional energy: E_phi(x | c) = -f_phi(x)[class_idx]."""
        logits = self.forward(x)
        return -logits[:, class_idx : class_idx + 1]

    def compute_bleed_penalty(self, x: Tensor, target_class: int) -> Tensor:
        """Computes cross-stem bleed penalty:
            L_bleed = log sum_{j != target} exp(f_phi(x)[j] - f_phi(x)[target])
        """
        logits = self.forward(x)
        target_logit = logits[:, target_class : target_class + 1]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, target_class] = False

        competitor_logits = logits.masked_select(mask).view(logits.size(0), -1)
        diff = competitor_logits - target_logit
        bleed_penalty = torch.logsumexp(diff, dim=-1, keepdim=True)
        return bleed_penalty

    def compute_conditional_score(self, x: Tensor, target_class: int) -> Tensor:
        """Computes analytical guidance score vector nab_x log p(c | x)."""
        assert x.requires_grad, "Input x must have requires_grad=True"
        logits = self.forward(x)
        log_prob = F.log_softmax(logits, dim=-1)[:, target_class].sum()
        score = torch.autograd.grad(
            outputs=log_prob,
            inputs=x,
            retain_graph=True,
            create_graph=False,
        )[0]
        return score


# ---------------------------------------------------------------------------
# 3. Persistent Replay Buffer for JEM MCMC Sampling
# ---------------------------------------------------------------------------


class JEMReplayBuffer:
    """Replay buffer storing negative waveform samples for JEM training."""

    def __init__(self, capacity: int = 300, reinit_prob: float = 0.05) -> None:
        self.capacity = capacity
        self.reinit_prob = reinit_prob
        self.buffer: list[Tensor] = []

    def sample(
        self,
        batch_size: int,
        audio_shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[Tensor, list[int]]:
        samples = []
        indices = []

        for _ in range(batch_size):
            if len(self.buffer) > 0 and random.random() > self.reinit_prob:
                idx = random.randint(0, len(self.buffer) - 1)
                samples.append(self.buffer[idx].clone().to(device))
                indices.append(idx)
            else:
                fresh = torch.randn(audio_shape, device=device) * 0.1
                samples.append(fresh)
                indices.append(-1)

        batch_tensor = torch.stack(samples, dim=0)
        return batch_tensor, indices

    def update(self, samples: Tensor, indices: list[int]) -> None:
        samples_cpu = samples.detach().cpu()
        for i, idx in enumerate(indices):
            if 0 <= idx < len(self.buffer):
                self.buffer[idx] = samples_cpu[i]
            else:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(samples_cpu[i])
                else:
                    r_idx = random.randint(0, len(self.buffer) - 1)
                    self.buffer[r_idx] = samples_cpu[i]


# ---------------------------------------------------------------------------
# 4. JEM Single-Source Separator with Bleed Barrier
# ---------------------------------------------------------------------------


class JEMSingleSourceSeparator:
    """Iterative Single-Source Separator using Joint Energy Model Guidance."""

    def __init__(
        self,
        jem: WaveformJEM,
        device: torch.device | None = None,
    ) -> None:
        self.jem = jem
        self.device = device or next(jem.parameters()).device
        self.jem.to(self.device)
        self.jem.eval()
        self.stft_loss_fn = MultiResolutionSTFTLoss().to(self.device)

    def separate(
        self,
        mix_audio: Tensor,
        target_class: int = 0,
        num_steps: int = 60,
        step_size: float = 0.04,
        lambda_prior: float = 0.5,
        lambda_bleed: float = 1.0,
        lambda_stft: float = 1.0,
        clamp_grad: float = 1.0,
        return_history: bool = True,
    ) -> dict[str, Any]:
        """Separates target stem from mixture via gradient descent on JEM energy landscape."""
        if mix_audio.ndim == 2:
            mix_audio = mix_audio.unsqueeze(1)
        mix_audio = mix_audio.to(self.device)
        batch_size, _channels, _length = mix_audio.shape

        target_stem = (mix_audio * 0.5).clone().detach().requires_grad_(True)

        history: dict[str, list[float]] = {
            "total_loss": [],
            "target_energy": [],
            "bleed_penalty": [],
            "consistency_loss": [],
        }

        optimizer = torch.optim.Adam([target_stem], lr=step_size)

        for _step in range(num_steps):
            optimizer.zero_grad()

            residual_stem = mix_audio - target_stem
            l2_consistency = F.mse_loss(target_stem + residual_stem, mix_audio)

            if lambda_stft > 0:
                stft_err = self.stft_loss_fn(target_stem, mix_audio)
            else:
                stft_err = torch.zeros(1, device=self.device)

            target_energy = self.jem.compute_class_energy(target_stem, target_class).mean()
            bleed_pen = self.jem.compute_bleed_penalty(target_stem, target_class).mean()

            total_loss = (
                l2_consistency
                + lambda_stft * stft_err
                + lambda_prior * target_energy
                + lambda_bleed * bleed_pen
            )

            total_loss.backward()
            if clamp_grad > 0:
                torch.nn.utils.clip_grad_norm_([target_stem], clamp_grad)
            optimizer.step()

            with torch.no_grad():
                target_stem.clamp_(-1.0, 1.0)

            if return_history:
                history["total_loss"].append(float(total_loss.item()))
                history["target_energy"].append(float(target_energy.item()))
                history["bleed_penalty"].append(float(bleed_pen.item()))
                history["consistency_loss"].append(float(l2_consistency.item()))

        with torch.no_grad():
            final_target = target_stem.detach()
            final_residual = mix_audio - final_target

        return {
            "target_stem": final_target,
            "residual_stem": final_residual,
            "history": history,
        }


# ---------------------------------------------------------------------------
# 5. JEM Diffusion & Generative Steering Bridge
# ---------------------------------------------------------------------------


class JEMGenerativeSteeringBridge:
    """Projects JEM acoustic priors into generative diffusion models (e.g. Stable Audio 3)."""

    def __init__(self, jem: WaveformJEM) -> None:
        self.jem = jem

    def get_latent_style_prior(self, audio: Tensor) -> Tensor:
        """Extracts penultimate acoustic style representation from JEM for AdaLN injection."""
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)

        h = audio
        for block in self.jem.down_blocks:
            h = block(h)

        h = self.jem.act_out(self.jem.conv_out(h))
        h_pool = 0.5 * (h.mean(dim=-1) + h.amax(dim=-1))
        return h_pool

    def compute_guidance_gradient(
        self,
        x_t: Tensor,
        target_class: int,
        guidance_scale: float = 2.0,
    ) -> Tensor:
        """Computes classifier-guidance gradient at diffusion step t."""
        score = self.jem.compute_conditional_score(x_t, target_class)
        return guidance_scale * score


__all__ = [
    "JEMGenerativeSteeringBridge",
    "JEMReplayBuffer",
    "JEMResidualBlock1D",
    "JEMSingleSourceSeparator",
    "WaveformJEM",
]
