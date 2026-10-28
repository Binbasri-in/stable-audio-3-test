"""Waveform Glow (Generative Flow with Invertible 1x1 Convolutions).

Production-grade Invertible Normalizing Flow for raw 1D acoustic waveforms,
exact log-likelihood evaluation, unconditional audio synthesis,
and Maximum A Posteriori (MAP) single-source separation inversion.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. 1D Squeeze and Unsqueeze (Temporal folding to Channel Dimension)
# ---------------------------------------------------------------------------


class Squeeze1D(nn.Module):
    """Folds adjacent temporal samples into channel dimensions.

    For a 1D waveform tensor [B, C_in, T], Squeeze1D transforms it to
    [B, C_in * factor, T // factor].

    Because this is a pure coordinate rearrangement (permutation of indices),
    the Jacobian determinant is identically 1, hence log |det J| = 0.
    """

    def __init__(self, factor: int = 8) -> None:
        super().__init__()
        if factor < 2:
            raise ValueError(f"Squeeze factor must be >= 2, got {factor}")
        self.factor = factor

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        """Squeeze waveform forward: [B, C, T] -> [B, C * factor, T // factor].

        Returns:
            squeezed: Tensor of shape [B, C * factor, T // factor]
            orig_len: Original temporal length T before potential padding
        """
        B, C, T = x.shape
        pad = (self.factor - (T % self.factor)) % self.factor
        if pad > 0:
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
            T_padded = T + pad
        else:
            T_padded = T

        L = T_padded // self.factor
        squeezed = (
            x.view(B, C, L, self.factor)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, C * self.factor, L)
        )
        return squeezed, T

    def inverse(self, x: Tensor, orig_len: int | None = None) -> Tensor:
        """Unsqueeze back to temporal domain: [B, C * factor, L] -> [B, C, T]."""
        B, C_squeezed, L = x.shape
        if C_squeezed % self.factor != 0:
            raise ValueError(
                f"Channel dimension {C_squeezed} must be divisible by squeeze factor {self.factor}"
            )
        C = C_squeezed // self.factor

        unsqueezed = (
            x.view(B, C, self.factor, L)
            .permute(0, 1, 3, 2)
            .contiguous()
            .view(B, C, L * self.factor)
        )

        if orig_len is not None and orig_len < unsqueezed.shape[-1]:
            unsqueezed = unsqueezed[..., :orig_len]

        return unsqueezed


# ---------------------------------------------------------------------------
# 2. 1D Activation Normalization (ActNorm1D)
# ---------------------------------------------------------------------------


class ActNorm1D(nn.Module):
    """Activation Normalization for 1D Temporal Channels (Kingma & Dhariwal, 2018).

    Performs an affine transformation on each channel:
        y = (x + bias) * exp(logs)

    The parameters (logs, bias) are initialized with data-dependent statistics
    on the first mini-batch so that each channel has zero mean and unit variance.

    Log-Determinant:
        log |det J| = L * sum_{c=1}^C logs_c
    """

    def __init__(self, channels: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def _initialize(self, x: Tensor) -> None:
        """Data-dependent initialization using the first mini-batch statistics."""
        with torch.no_grad():
            mean = x.mean(dim=[0, 2], keepdim=True)
            std = x.std(dim=[0, 2], keepdim=True) + self.epsilon

            self.bias.data.copy_(-mean)
            self.logs.data.copy_(-torch.log(std))
            self.initialized.fill_(1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward affine scaling: x -> y.

        Args:
            x: Tensor of shape [B, C, L]

        Returns:
            y: Scaled tensor of shape [B, C, L]
            log_det: Tensor of shape [B] representing log |det J|
        """
        B, C, L = x.shape
        if not self.initialized:
            self._initialize(x)

        y = (x + self.bias) * torch.exp(self.logs)
        log_det = L * torch.sum(self.logs)
        log_det = log_det.expand(B)
        return y, log_det

    def inverse(self, y: Tensor) -> Tensor:
        """Inverse affine scaling: y -> x."""
        return y * torch.exp(-self.logs) - self.bias


# ---------------------------------------------------------------------------
# 3. Invertible 1x1 Convolution with LU Parameterization
# ---------------------------------------------------------------------------


class Invertible1x1Conv1D(nn.Module):
    """Invertible 1x1 1D Convolution with LU Decomposition (Glow / WaveGlow).

    Parameterizes the weight matrix W in R^{C x C} via LU decomposition:
        W = P * L * (U + diag(s))

    Exact Log-Determinant:
        log |det J| = L * sum_{c=1}^C log |s_c|
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        # 1. Sample a random orthogonal matrix via QR decomposition
        w_init = torch.linalg.qr(torch.randn(channels, channels))[0]

        # 2. Compute LU decomposition of the initial matrix
        p_mat, l_mat, u_mat = torch.linalg.lu(w_init)

        s_diag = torch.diag(u_mat)
        s_sign = torch.sign(s_diag)
        s_sign[s_sign == 0] = 1.0
        log_s = torch.log(torch.abs(s_diag) + 1e-6)

        u_triu = torch.triu(u_mat, diagonal=1)
        l_tril = torch.tril(l_mat, diagonal=-1)

        self.register_buffer("p_mat", p_mat)
        self.register_buffer("s_sign", s_sign)
        self.register_buffer("l_mask", torch.tril(torch.ones(channels, channels), diagonal=-1))
        self.register_buffer("u_mask", torch.triu(torch.ones(channels, channels), diagonal=1))
        self.register_buffer("eye", torch.eye(channels))

        self.l_weight = nn.Parameter(l_tril)
        self.u_weight = nn.Parameter(u_triu)
        self.log_s = nn.Parameter(log_s)

    def _assemble_weights(self) -> tuple[Tensor, Tensor]:
        L = self.l_weight * self.l_mask + self.eye
        s = self.s_sign * torch.exp(self.log_s)
        U = self.u_weight * self.u_mask + torch.diag(s)
        W = self.p_mat @ L @ U
        W_inv = torch.inverse(W)
        return W, W_inv

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, C, L = x.shape
        W, _ = self._assemble_weights()
        y = torch.matmul(W, x)
        log_det = L * torch.sum(self.log_s)
        log_det = log_det.expand(B)
        return y, log_det

    def inverse(self, y: Tensor) -> Tensor:
        _, W_inv = self._assemble_weights()
        x = torch.matmul(W_inv, y)
        return x


# ---------------------------------------------------------------------------
# 4. Gated Dilated WaveNet Residual Layer & Affine Coupling Block 1D
# ---------------------------------------------------------------------------


class DilatedResidualLayer1D(nn.Module):
    """Gated 1D Dilated Convolutional Residual Layer (WaveNet Style)."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = (kernel_size - 1) * dilation // 2

        self.dilated_conv = nn.Conv1d(
            hidden_dim,
            2 * hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.res_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.skip_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.dilated_conv(x)
        h_filter, h_gate = torch.chunk(h, 2, dim=1)
        gated = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(gated) + x
        skip = self.skip_conv(gated)
        return res, skip


class WaveNetCouplingBlock1D(nn.Module):
    """Affine Coupling Layer conditioned on a Gated Dilated 1D ConvNet."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        scale_bound: float = 3.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.in_channels_a = channels // 2
        self.in_channels_b = channels - self.in_channels_a
        self.scale_bound = scale_bound

        self.in_proj = nn.Conv1d(self.in_channels_a, hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer1D(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**i,
                )
                for i in range(num_layers)
            ]
        )
        self.out_proj = nn.Conv1d(hidden_dim, 2 * self.in_channels_b, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _transform(self, x_a: Tensor) -> tuple[Tensor, Tensor]:
        h = self.in_proj(x_a)
        skip_total = torch.zeros_like(h)

        for layer in self.layers:
            h, skip = layer(h)
            skip_total = skip_total + skip

        out = self.out_proj(skip_total)
        log_s, t = torch.chunk(out, 2, dim=1)
        log_s = self.scale_bound * torch.tanh(log_s / self.scale_bound)
        return log_s, t

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, C, L = x.shape
        x_a = x[:, : self.in_channels_a, :]
        x_b = x[:, self.in_channels_a :, :]

        log_s, t = self._transform(x_a)
        y_a = x_a
        y_b = x_b * torch.exp(log_s) + t

        y = torch.cat([y_a, y_b], dim=1)
        log_det = torch.sum(log_s, dim=[1, 2])
        return y, log_det

    def inverse(self, y: Tensor) -> Tensor:
        y_a = y[:, : self.in_channels_a, :]
        y_b = y[:, self.in_channels_a :, :]

        log_s, t = self._transform(y_a)
        x_a = y_a
        x_b = (y_b - t) * torch.exp(-log_s)

        x = torch.cat([x_a, x_b], dim=1)
        return x


# ---------------------------------------------------------------------------
# 5. Full Invertible Flow Step
# ---------------------------------------------------------------------------


class GlowFlowStep1D(nn.Module):
    """Complete Invertible Flow Step (ActNorm + Invertible 1x1 Conv + Affine Coupling)."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        scale_bound: float = 3.0,
    ) -> None:
        super().__init__()
        self.actnorm = ActNorm1D(channels=channels)
        self.invconv = Invertible1x1Conv1D(channels=channels)
        self.coupling = WaveNetCouplingBlock1D(
            channels=channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            scale_bound=scale_bound,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x, ld1 = self.actnorm(x)
        x, ld2 = self.invconv(x)
        y, ld3 = self.coupling(x)
        total_ld = ld1 + ld2 + ld3
        return y, total_ld

    def inverse(self, y: Tensor) -> Tensor:
        x = self.coupling.inverse(y)
        x = self.invconv.inverse(x)
        x = self.actnorm.inverse(x)
        return x


# ---------------------------------------------------------------------------
# 6. WaveformGlow: End-to-End Invertible Normalizing Flow Model
# ---------------------------------------------------------------------------


class WaveformGlow(nn.Module):
    """Waveform Glow: Generative Invertible Flow for Raw 1D Audio Waveforms."""

    def __init__(
        self,
        in_channels: int = 1,
        squeeze_factor: int = 16,
        num_steps: int = 8,
        hidden_dim: int = 64,
        num_layers_per_step: int = 4,
        scale_bound: float = 3.0,
        sample_rate: int = 44100,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_factor = squeeze_factor
        self.flow_channels = in_channels * squeeze_factor
        self.num_steps = num_steps
        self.sample_rate = sample_rate

        self.squeeze = Squeeze1D(factor=squeeze_factor)
        self.steps = nn.ModuleList(
            [
                GlowFlowStep1D(
                    channels=self.flow_channels,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers_per_step,
                    scale_bound=scale_bound,
                )
                for _ in range(num_steps)
            ]
        )

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        B, C, T = waveform.shape
        x, orig_len = self.squeeze(waveform)

        total_log_det = torch.zeros(B, device=waveform.device)
        for step in self.steps:
            x, step_log_det = step(x)
            total_log_det = total_log_det + step_log_det

        return x, total_log_det

    def inverse(self, z: Tensor, orig_len: int | None = None) -> Tensor:
        x = z
        for step in reversed(self.steps):
            x = step.inverse(x)

        waveform = self.squeeze.inverse(x, orig_len=orig_len)
        return waveform

    def log_prob(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        B, C, T = waveform.shape
        z, log_det = self.forward(waveform)

        D = z.shape[1] * z.shape[2]
        log_pz = -0.5 * torch.sum(z**2, dim=[1, 2]) - 0.5 * D * math.log(2.0 * math.pi)
        log_p = log_pz + log_det
        bpd = -log_p / (T * math.log(2.0))
        return log_p, bpd

    def loss(self, waveform: Tensor) -> Tensor:
        _, bpd = self.log_prob(waveform)
        return bpd.mean()

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 1,
        length: int = 44100,
        temperature: float = 0.8,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        self.eval()
        pad = (self.squeeze_factor - (length % self.squeeze_factor)) % self.squeeze_factor
        total_len = length + pad
        L = total_len // self.squeeze_factor

        z = torch.randn(num_samples, self.flow_channels, L, device=device) * temperature
        samples = self.inverse(z, orig_len=length)
        return samples


# ---------------------------------------------------------------------------
# 7. Maximum A Posteriori (MAP) Single-Source Separation Solver
# ---------------------------------------------------------------------------


class GlowMAPSeparator:
    """Maximum A Posteriori (MAP) Single-Source Separator using Exact Glow Priors."""

    def __init__(
        self,
        glow_model: WaveformGlow,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = glow_model
        self.device = device or next(glow_model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    def separate_latent(
        self,
        mixture: Tensor,
        num_iterations: int = 100,
        lr: float = 0.02,
        sigma_mix: float = 0.05,
        lambda_prior: float = 0.5,
        target_energy_ratio: float = 0.6,
        verbose: bool = False,
    ) -> dict[str, Any]:
        mixture = mixture.to(self.device)
        B, C, T = mixture.shape

        with torch.no_grad():
            z_init, _ = self.model.forward(mixture * target_energy_ratio)
            z = z_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=lr)
        history = {
            "loss": [],
            "fidelity_loss": [],
            "prior_loss": [],
        }

        for i in range(num_iterations):
            optimizer.zero_grad()
            s_hat = self.model.inverse(z, orig_len=T)
            s_res = mixture - s_hat

            fidelity_loss = F.mse_loss(s_hat + s_res, mixture) / (2.0 * sigma_mix**2)
            energy_penalty = F.relu(torch.mean(s_hat**2) - torch.mean(mixture**2))
            prior_loss = 0.5 * torch.mean(z**2)

            total_loss = fidelity_loss + lambda_prior * prior_loss + 10.0 * energy_penalty
            total_loss.backward()
            optimizer.step()

            history["loss"].append(total_loss.item())
            history["fidelity_loss"].append(fidelity_loss.item())
            history["prior_loss"].append(prior_loss.item())

            if verbose and (i % 25 == 0 or i == num_iterations - 1):
                print(f"[Latent MAP Iter {i:03d}] Loss: {total_loss.item():.4f} | Prior: {prior_loss.item():.4f}")

        with torch.no_grad():
            final_target = self.model.inverse(z, orig_len=T).detach()
            final_residual = (mixture - final_target).detach()

        return {
            "target": final_target,
            "residual": final_residual,
            "latent_z": z.detach(),
            "history": history,
            "method": "latent_space_map",
        }

    def separate_waveform(
        self,
        mixture: Tensor,
        num_iterations: int = 100,
        lr: float = 0.005,
        sigma_mix: float = 0.05,
        lambda_prior: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, Any]:
        mixture = mixture.to(self.device)
        B, C, T = mixture.shape

        s = (mixture * 0.5).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([s], lr=lr)

        history = {
            "loss": [],
            "fidelity_loss": [],
            "log_prob": [],
        }

        for i in range(num_iterations):
            optimizer.zero_grad()
            s_res = mixture - s
            fidelity_loss = F.mse_loss(s + s_res, mixture) / (2.0 * sigma_mix**2)
            energy_penalty = F.relu(torch.mean(s**2) - torch.mean(mixture**2))

            log_p, _ = self.model.log_prob(s)
            prior_loss = -torch.mean(log_p)

            total_loss = fidelity_loss + lambda_prior * prior_loss + 10.0 * energy_penalty
            total_loss.backward()
            optimizer.step()

            history["loss"].append(total_loss.item())
            history["fidelity_loss"].append(fidelity_loss.item())
            history["log_prob"].append(log_p.mean().item())

            if verbose and (i % 25 == 0 or i == num_iterations - 1):
                print(f"[Waveform MAP Iter {i:03d}] Loss: {total_loss.item():.4f} | Log-Prob: {log_p.mean().item():.2f}")

        final_target = s.detach()
        final_residual = (mixture - final_target).detach()

        return {
            "target": final_target,
            "residual": final_residual,
            "history": history,
            "method": "waveform_gradient_ascent",
        }


__all__ = [
    "ActNorm1D",
    "DilatedResidualLayer1D",
    "GlowFlowStep1D",
    "GlowMAPSeparator",
    "Invertible1x1Conv1D",
    "Squeeze1D",
    "WaveNetCouplingBlock1D",
    "WaveformGlow",
]
