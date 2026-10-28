"""Compositional Latent Energy-Based Models (EBM) for Source Separation.

Implements explicit scalar Energy-Based Models (EBMs) and Langevin MCMC
dynamics for source-only supervised music source separation:
  1. SmoothResidualBlock1D: Continuous gradient residual block with SiLU activations.
  2. ScalarLatentEBM: Deep scalar energy function mapping compressed latents to scalar energy.
  3. EBMReplayBuffer: Persistent negative sample buffer for Contrastive Divergence training.
  4. sample_langevin_mcmc: Short-run SGLD MCMC dynamics sampler.
  5. CompositionalLangevinSeparator: Test-time product-of-experts energy composition and SGLD demixing.
"""

from __future__ import annotations

import math
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
# 1. Smooth Residual 1D Block for Energy Networks
# ---------------------------------------------------------------------------


class SmoothResidualBlock1D(nn.Module):
    """Residual block with SiLU activations for continuous energy gradients."""

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
# 2. Scalar Latent Energy-Based Model Architecture
# ---------------------------------------------------------------------------


class ScalarLatentEBM(nn.Module):
    """Explicit Scalar Energy-Based Model E_theta(z) over continuous audio latents."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Input projection
        self.in_conv = nn.Conv1d(
            in_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.in_act = nn.SiLU()

        # Residual backbone with progressive dilations
        self.res_blocks = nn.ModuleList()
        dilations = [1, 2, 4, 8][:num_layers]
        while len(dilations) < num_layers:
            dilations.append(1)

        for d in dilations:
            self.res_blocks.append(
                SmoothResidualBlock1D(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                )
            )

        # Multi-scale pooling and scalar output head
        self.out_conv1 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.out_act = nn.SiLU()
        self.out_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, z: Tensor) -> Tensor:
        """Evaluates scalar energy E_theta(z) for each item in the batch."""
        if z.ndim == 2:
            z = z.unsqueeze(-1)  # [B, C, 1]

        h = self.in_act(self.in_conv(z))
        for block in self.res_blocks:
            h = block(h)

        h = self.out_act(self.out_conv1(h))  # [B, hidden_dim // 2, L]

        # Global average + max pooling for rich temporal feature aggregation
        h_avg = h.mean(dim=-1)  # [B, hidden_dim // 2]
        h_max = h.amax(dim=-1)  # [B, hidden_dim // 2]
        h_pool = 0.5 * (h_avg + h_max)

        energy = self.out_head(h_pool)  # [B, 1]
        return energy

    def compute_energy_gradient(
        self,
        z: Tensor,
        create_graph: bool = False,
    ) -> Tensor:
        """Computes grad_z E_theta(z) analytically via PyTorch autograd."""
        assert z.requires_grad, "Input latent z must have requires_grad=True"
        energy = self.forward(z)
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=z,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad


# ---------------------------------------------------------------------------
# 3. Persistent MCMC Replay Buffer for Contrastive Divergence
# ---------------------------------------------------------------------------


class EBMReplayBuffer:
    """Fixed-capacity replay buffer storing negative samples for Contrastive Divergence."""

    def __init__(
        self,
        capacity: int = 500,
        reinit_prob: float = 0.05,
    ) -> None:
        self.capacity = capacity
        self.reinit_prob = reinit_prob
        self.buffer: list[Tensor] = []

    def sample(
        self,
        batch_size: int,
        latent_shape: tuple[int, ...],
        device: torch.device,
    ) -> tuple[Tensor, list[int]]:
        """Samples a batch of negative latent candidates from buffer or prior."""
        samples = []
        indices = []

        for _ in range(batch_size):
            if len(self.buffer) > 0 and random.random() > self.reinit_prob:
                idx = random.randint(0, len(self.buffer) - 1)
                samples.append(self.buffer[idx].clone().to(device))
                indices.append(idx)
            else:
                fresh = torch.randn(latent_shape, device=device)
                samples.append(fresh)
                indices.append(-1)

        batch_tensor = torch.stack(samples, dim=0)
        return batch_tensor, indices

    def update(self, samples: Tensor, indices: list[int]) -> None:
        """Pushes updated MCMC samples back into the buffer."""
        samples_cpu = samples.detach().cpu()
        for i, idx in enumerate(indices):
            if idx >= 0 and idx < len(self.buffer):
                self.buffer[idx] = samples_cpu[i]
            else:
                if len(self.buffer) < self.capacity:
                    self.buffer.append(samples_cpu[i])
                else:
                    replace_idx = random.randint(0, len(self.buffer) - 1)
                    self.buffer[replace_idx] = samples_cpu[i]


# ---------------------------------------------------------------------------
# 4. Short-Run Langevin Dynamics Sampler
# ---------------------------------------------------------------------------


def sample_langevin_mcmc(
    ebm: ScalarLatentEBM,
    init_z: Tensor,
    num_steps: int = 30,
    step_size: float = 0.01,
    noise_scale: float = 0.005,
    clamp_grad: float = 1.0,
) -> Tensor:
    """Performs short-run Stochastic Gradient Langevin Dynamics (SGLD) to draw samples."""
    z = init_z.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        energy = ebm(z)
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=z,
            retain_graph=False,
            create_graph=False,
        )[0]

        if clamp_grad > 0:
            grad = torch.clamp(grad, -clamp_grad, clamp_grad)

        noise = torch.randn_like(z) if noise_scale > 0 else 0.0
        z.data = z.data - 0.5 * step_size * grad + math.sqrt(step_size) * noise_scale * noise

    return z.detach()


# ---------------------------------------------------------------------------
# 5. Compositional Langevin Separator (Test-Time Product of Experts)
# ---------------------------------------------------------------------------


class CompositionalLangevinSeparator:
    """Compositional Source Separation via Energy-Based Langevin Dynamics."""

    def __init__(
        self,
        energy_models: dict[str, ScalarLatentEBM],
        decoders: dict[str, nn.Module],
        device: torch.device | None = None,
    ) -> None:
        self.source_names = list(energy_models.keys())
        self.energy_models = energy_models
        self.decoders = decoders

        if device is None:
            self.device = next(iter(energy_models.values())).parameters().__next__().device
        else:
            self.device = device

        for model in self.energy_models.values():
            if isinstance(model, nn.Module):
                model.eval()
                model.to(self.device)
        for dec in self.decoders.values():
            if isinstance(dec, nn.Module):
                dec.eval()
                dec.to(self.device)

        self.stft_loss_fn = MultiResolutionSTFTLoss().to(self.device)

    def _decode_source(self, name: str, latent: Tensor, target_len: int) -> Tensor:
        dec = self.decoders[name]
        if hasattr(dec, "decode") and callable(dec.decode):
            return dec.decode(latent, target_len=target_len)
        return dec(latent, target_len=target_len)

    def separate(
        self,
        mix_audio: Tensor,
        initial_latents: dict[str, Tensor] | None = None,
        latent_shape: tuple[int, int] = (64, 258),
        num_steps: int = 50,
        initial_step_size: float = 0.05,
        final_step_size: float = 0.005,
        noise_scale: float = 0.005,
        lambda_mix: float = 15.0,
        lambda_stft: float = 2.0,
        clamp_grad: float = 1.0,
        return_trajectories: bool = True,
    ) -> dict[str, Any]:
        """Separates an input audio mixture into component sources via SGLD."""
        if mix_audio.ndim == 2:
            mix_audio = mix_audio.unsqueeze(1)
        mix_audio = mix_audio.to(self.device)
        batch_size, _, target_len = mix_audio.shape

        # 1. Initialize latent vectors
        latents: dict[str, Tensor] = {}
        for name in self.source_names:
            if initial_latents is not None and name in initial_latents:
                latents[name] = initial_latents[name].clone().to(self.device)
            else:
                latents[name] = torch.randn(
                    batch_size,
                    latent_shape[0],
                    latent_shape[1],
                    device=self.device,
                ) * 0.1
            latents[name].requires_grad_(True)

        history: dict[str, list[float]] = {
            "total_energy": [],
            "prior_energy": [],
            "consistency_loss": [],
            "stft_loss": [],
        }
        for name in self.source_names:
            history[f"energy_{name}"] = []

        # 2. Annealed SGLD Optimization Loop
        for step in range(num_steps):
            progress = step / max(1, num_steps - 1)
            step_size = final_step_size + 0.5 * (initial_step_size - final_step_size) * (
                1.0 + math.cos(math.pi * progress)
            )

            # Prior energy
            total_prior_energy = torch.zeros(1, device=self.device)
            prior_energies: dict[str, Tensor] = {}
            for name in self.source_names:
                e_k = self.energy_models[name](latents[name]).mean()
                prior_energies[name] = e_k
                total_prior_energy = total_prior_energy + e_k

            # Decode
            decoded_sources: dict[str, Tensor] = {}
            reconstructed_mix = torch.zeros_like(mix_audio)
            for name in self.source_names:
                s_hat = self._decode_source(name, latents[name], target_len=target_len)
                decoded_sources[name] = s_hat
                reconstructed_mix = reconstructed_mix + s_hat

            mix_l2 = F.mse_loss(reconstructed_mix, mix_audio)

            if lambda_stft > 0:
                stft_err = self.stft_loss_fn(reconstructed_mix, mix_audio)
            else:
                stft_err = torch.zeros(1, device=self.device)

            total_energy = total_prior_energy + lambda_mix * mix_l2 + lambda_stft * stft_err

            grads = torch.autograd.grad(
                outputs=total_energy,
                inputs=list(latents.values()),
                retain_graph=False,
                create_graph=False,
            )

            with torch.no_grad():
                for idx, name in enumerate(self.source_names):
                    g = grads[idx]
                    if clamp_grad > 0:
                        g = torch.clamp(g, -clamp_grad, clamp_grad)

                    cur_noise = (
                        math.sqrt(step_size)
                        * noise_scale
                        * (1.0 - 0.7 * progress)
                        * torch.randn_like(latents[name])
                    )
                    latents[name].data = latents[name].data - 0.5 * step_size * g + cur_noise

            if return_trajectories:
                history["total_energy"].append(float(total_energy.item()))
                history["prior_energy"].append(float(total_prior_energy.item()))
                history["consistency_loss"].append(float(mix_l2.item()))
                history["stft_loss"].append(float(stft_err.item()))
                for name in self.source_names:
                    history[f"energy_{name}"].append(float(prior_energies[name].item()))

        final_stems: dict[str, Tensor] = {}
        with torch.no_grad():
            for name in self.source_names:
                final_stems[name] = self._decode_source(name, latents[name], target_len=target_len)

        return {
            "stems": final_stems,
            "final_latents": {k: v.detach() for k, v in latents.items()},
            "reconstructed_mix": sum(final_stems.values()),
            "history": history,
        }


__all__ = [
    "CompositionalLangevinSeparator",
    "EBMReplayBuffer",
    "ScalarLatentEBM",
    "SmoothResidualBlock1D",
    "sample_langevin_mcmc",
]
