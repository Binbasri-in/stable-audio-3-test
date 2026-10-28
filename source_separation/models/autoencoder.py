"""Single Source Separation Waveform & Spectrogram Autoencoder Architectures.

Provides modular autoencoder architectures for raw waveforms and 2D spectrograms:
  1. WaveformAutoencoder: Direct 1D time-domain convolutional encoder-decoder with
     temporal striding, dilated residual blocks, compact bottleneck z, and
     transposed convolutional upsampling back to exact sample length.
  2. WaveformUNetAutoencoder: Waveform autoencoder with U-Net skip connections
     for high-fidelity transient and phase reconstruction.
  3. ResidualBlock1D, DownsampleBlock1D, UpsampleBlock1D: 1D convolutional blocks.
  4. SingleSourceAutoencoder, SpectrogramEncoder, SpectrogramDecoder: 2D spectrogram autoencoders.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import 2D spectrogram architectures for unified availability
from .spectrogram import (
    DownsampleBlock2D,
    ResidualBlock2D,
    SingleSourceAutoencoder,
    SpectrogramDecoder,
    SpectrogramEncoder,
    UpsampleBlock2D,
)

# ---------------------------------------------------------------------------
# 1. 1D Waveform Convolutional Blocks
# ---------------------------------------------------------------------------


class ResidualBlock1D(nn.Module):
    """1D Dilated Residual Convolutional Block for raw waveform processing."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
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
            bias=False,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        res = x
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + res)


class DownsampleBlock1D(nn.Module):
    """Strided 1D convolution block for temporal downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = stride * 2
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res = ResidualBlock1D(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm(self.conv(x)))
        return self.res(h)


class UpsampleBlock1D(nn.Module):
    """Transposed 1D convolution block for temporal upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 4,
        kernel_size: int | None = None,
    ) -> None:
        super().__init__()
        if kernel_size is None:
            kernel_size = stride * 2
        padding = kernel_size // 2
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res = ResidualBlock1D(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm(self.deconv(x)))
        return self.res(h)


# ---------------------------------------------------------------------------
# 2. Waveform Autoencoder Architecture
# ---------------------------------------------------------------------------


class WaveformAutoencoder(nn.Module):
    """Time-Domain Waveform Autoencoder for Single Source Separation.

    Takes a mixture waveform [B, in_channels, T], extracts a compact acoustic
    bottleneck representation z in [B, latent_dim, T_latent], and reconstructs
    the target single source waveform [B, out_channels, T].

    Operates directly on physical pressure oscillations to eliminate phase
    estimation errors and preserve exact linear superposition.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int | None = None,
        base_channels: int = 32,
        latent_dim: int = 128,
        strides: tuple[int, ...] = (4, 4, 4, 4),
        max_channels: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.strides = strides
        self.max_channels = max_channels

        # Build dynamic encoder channel progression across strides
        enc_channels = [self.in_channels]
        curr = base_channels
        for _ in range(len(strides)):
            enc_channels.append(min(curr, max_channels))
            curr = curr * 2

        self.encoder_blocks = nn.ModuleList([
            DownsampleBlock1D(enc_channels[i], enc_channels[i + 1], stride=strides[i])
            for i in range(len(strides))
        ])

        last_enc = enc_channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv1d(last_enc, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder: Latent projection and transposed upsampling stages
        self.dec_proj = nn.Sequential(
            nn.Conv1d(latent_dim, last_enc, kernel_size=3, padding=1),
            nn.BatchNorm1d(last_enc),
            nn.LeakyReLU(0.2, inplace=True),
        )

        dec_channels = list(reversed(enc_channels[1:]))
        rev_strides = list(reversed(strides))
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(rev_strides) - 1):
            self.decoder_blocks.append(
                UpsampleBlock1D(dec_channels[i], dec_channels[i + 1], stride=rev_strides[i])
            )

        final_stride = rev_strides[-1]
        self.final_deconv = nn.ConvTranspose1d(
            dec_channels[-1],
            self.out_channels,
            kernel_size=final_stride * 2,
            stride=final_stride,
            padding=final_stride // 2,
            bias=True,
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encodes mixture waveform into latent representation z."""
        h = x
        for blk in self.encoder_blocks:
            h = blk(h)
        return self.bottleneck(h)

    def decode(self, z: Tensor, target_len: int | None = None) -> Tensor:
        """Decodes latent vector z into reconstructed source waveform."""
        h = self.dec_proj(z)
        for blk in self.decoder_blocks:
            h = blk(h)
        out = self.final_deconv(h)

        # Ensure exact sample length alignment
        if target_len is not None:
            curr_len = out.shape[-1]
            if curr_len > target_len:
                out = out[..., :target_len]
            elif curr_len < target_len:
                out = F.pad(out, (0, target_len - curr_len))

        return torch.tanh(out)

    def forward(self, mix: Tensor) -> dict[str, Tensor]:
        """Forward pass taking input mixture waveform.

        Args:
            mix: [B, C, T] mixture audio tensor.

        Returns:
            Dict containing:
                'recon': [B, C, T] reconstructed target stem waveform.
                'latent': [B, latent_dim, T_latent] latent bottleneck feature.
                'residual': [B, C, T] estimated residual accompaniment.
        """
        target_len = mix.shape[-1]
        z = self.encode(mix)
        recon = self.decode(z, target_len=target_len)
        residual = mix - recon

        return {
            "recon": recon,
            "latent": z,
            "residual": residual,
        }


# ---------------------------------------------------------------------------
# 3. Waveform U-Net Autoencoder (with Skip Connections)
# ---------------------------------------------------------------------------


class WaveformUNetAutoencoder(nn.Module):
    """Waveform Autoencoder with U-Net Skip Connections.

    Combines temporal hierarchical bottleneck compression with high-resolution
    skip connections from encoder levels to decoder levels for sharp transient recovery.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.enc1 = DownsampleBlock1D(in_channels, base_channels, stride=4)
        self.enc2 = DownsampleBlock1D(base_channels, base_channels * 2, stride=4)
        self.enc3 = DownsampleBlock1D(base_channels * 2, base_channels * 4, stride=4)
        self.enc4 = DownsampleBlock1D(base_channels * 4, base_channels * 8, stride=4)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_channels * 8, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec_proj = nn.Sequential(
            nn.Conv1d(latent_dim, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoders concatenate skip features: in_channels * 2
        self.dec4 = UpsampleBlock1D(base_channels * 8 * 2, base_channels * 4, stride=4)
        self.dec3 = UpsampleBlock1D(base_channels * 4 * 2, base_channels * 2, stride=4)
        self.dec2 = UpsampleBlock1D(base_channels * 2 * 2, base_channels, stride=4)
        self.dec1 = nn.ConvTranspose1d(
            base_channels * 2, in_channels, kernel_size=8, stride=4, padding=4, bias=True
        )

    def forward(self, mix: Tensor) -> dict[str, Tensor]:
        target_len = mix.shape[-1]

        # Encoder with skip features
        e1 = self.enc1(mix)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z = self.bottleneck(e4)
        d4_in = self.dec_proj(z)

        # Decoder with U-Net skips
        d4_cat = self._cat_skip(d4_in, e4)
        d3 = self.dec4(d4_cat)

        d3_cat = self._cat_skip(d3, e3)
        d2 = self.dec3(d3_cat)

        d2_cat = self._cat_skip(d2, e2)
        d1 = self.dec2(d2_cat)

        d1_cat = self._cat_skip(d1, e1)
        out = self.dec1(d1_cat)

        if out.shape[-1] > target_len:
            out = out[..., :target_len]
        elif out.shape[-1] < target_len:
            out = F.pad(out, (0, target_len - out.shape[-1]))

        recon = torch.tanh(out)
        return {
            "recon": recon,
            "latent": z,
            "residual": mix - recon,
        }

    @staticmethod
    def _cat_skip(d: Tensor, e: Tensor) -> Tensor:
        min_len = min(d.shape[-1], e.shape[-1])
        return torch.cat([d[..., :min_len], e[..., :min_len]], dim=1)


__all__ = [
    # 1D Waveform Blocks & Models
    "ResidualBlock1D",
    "DownsampleBlock1D",
    "UpsampleBlock1D",
    "WaveformAutoencoder",
    "WaveformUNetAutoencoder",
    # 2D Spectrogram Blocks & Models
    "ResidualBlock2D",
    "DownsampleBlock2D",
    "UpsampleBlock2D",
    "SpectrogramEncoder",
    "SpectrogramDecoder",
    "SingleSourceAutoencoder",
]
