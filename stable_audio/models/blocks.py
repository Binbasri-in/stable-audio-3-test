import math
from functools import reduce
from typing import Callable, Dict, List, Literal, Optional

import numpy as np
import torch
from alias_free_torch import Activation1d
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from .utils import compile


def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * (torch.sin(x * alpha) ** 2)


class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale

        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return snake_beta(x, alpha, beta)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=16.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, t):
        in_dtype = t.dtype
        t = t.float()

        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        ramp = torch.linspace(0, 1, half_dim, device=t.device, dtype=torch.float32)
        log_min = math.log(self.min_freq)
        log_max = math.log(self.max_freq)

        freqs = torch.exp(ramp * (log_max - log_min) + log_min)
        args = t * freqs * 2 * math.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        return embedding.to(in_dtype)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False, depthwise=False, bias=True):
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
                groups=1 if not depthwise else out_channels,
                bias=bias,
            ),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        return x + self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False, kernel_size=5, conv_bias=True):
        skip = None if c_in == c_out else nn.Conv1d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv1d(c_in, c_mid, kernel_size, padding=kernel_size // 2, bias=conv_bias),
                nn.GroupNorm(1, c_mid),
                nn.GELU(),
                nn.Conv1d(c_mid, c_out, kernel_size, padding=kernel_size // 2, bias=conv_bias),
                nn.GroupNorm(1, c_out) if not is_last else nn.Identity(),
                nn.GELU() if not is_last else nn.Identity(),
            ],
            skip,
        )


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        return rearrange(x, "... a b -> ... b a")


def _zero_pad_modulo_sequence(x, size, dim=-2):
    input_len = x.shape[dim]
    pad_len = (size - input_len % size) % size
    if pad_len > 0:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        x = torch.cat([x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=dim)
    return x


class ResidualDownsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, conv_bias=True, **kwargs):
        super().__init__()
        self.scale_factor = 1 / stride
        self.layer = WNConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2) if stride > 1 else 0,
            bias=conv_bias,
        )
        self.map = WNConv1d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.scale_factor != 1:
            res = F.interpolate(x, scale_factor=self.scale_factor, mode="linear")
            filtered = self.layer(x)
            out = (1.0 / math.sqrt(2)) * (self.map(res) + filtered)
        else:
            out = self.map(x)
        return out


class ResidualUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride, conv_bias=True, causal=False, **kwargs):
        super().__init__()
        self.scale_factor = stride
        self.layer = WNConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2) if stride > 1 else 0,
            bias=conv_bias,
        )
        self.map = WNConv1d(in_channels, out_channels, 1)

    def forward(self, x):
        if self.scale_factor != 1:
            res = F.interpolate(x, scale_factor=self.scale_factor, mode="linear")
            filtered = self.layer(x)
            out = (1.0 / math.sqrt(2)) * (self.map(res) + filtered)
        else:
            out = self.map(x)
        return out
