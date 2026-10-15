import torch
import torch.nn as nn
from einops import rearrange
from torchaudio.transforms import Resample

from .blocks import ResidualUnit, WNConv1d


class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()
        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None
        self.enable_grad = enable_grad

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError


class AutoencoderPretransform(Pretransform):
    def __init__(self, model, scale=1.0, model_half=False, iterate_batch=False, chunked=False, enable_grad=False):
        super().__init__(
            enable_grad=enable_grad,
            io_channels=model.io_channels,
            is_discrete=model.bottleneck is not None and model.bottleneck.is_discrete,
        )
        self.model = model
        if not enable_grad:
            self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.sample_rate = model.sample_rate

        self.model_half = model_half
        self.iterate_batch = iterate_batch
        self.encoded_channels = model.latent_dim
        self.chunked = chunked
        self.num_quantizers = (
            model.bottleneck.num_quantizers if model.bottleneck is not None and model.bottleneck.is_discrete else None
        )
        self.codebook_size = (
            model.bottleneck.codebook_size if model.bottleneck is not None and model.bottleneck.is_discrete else None
        )

        if self.model_half:
            self.model.half()

    def encode(self, x, **kwargs):
        if self.model_half:
            x = x.half()
            self.model.to(torch.float16)

        encoded = self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            encoded = encoded.float()

        return encoded / self.scale

    def decode(self, z, **kwargs):
        z = z * self.scale

        if self.model_half:
            z = z.half()
            self.model.to(torch.float16)

        decoded = self.model.decode_audio(z, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs)

        if self.model_half:
            decoded = decoded.float()

        return decoded


class PatchedPretransform(Pretransform):
    def __init__(self, channels, patch_size, oversampling=1, postfilter_channels=0, **kwargs):
        super().__init__(enable_grad=True, io_channels=channels, is_discrete=False)
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling

        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size

        if self.oversampling > 1:
            self.input_upsampler = Resample(1, self.oversampling)
            self.output_downsampler = Resample(self.oversampling, 1)

        if postfilter_channels > 0:
            self.postfilter = nn.Sequential(
                WNConv1d(in_channels=channels, out_channels=postfilter_channels, kernel_size=7, padding=3, bias=True),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=1, use_snake=True, bias=True),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=3, use_snake=True, bias=True),
                ResidualUnit(in_channels=postfilter_channels, out_channels=postfilter_channels, dilation=9, use_snake=True, bias=True),
                WNConv1d(in_channels=postfilter_channels, out_channels=channels, kernel_size=7, padding=3, bias=False),
            )

    def _pad(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x

    def encode(self, x):
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        x = rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)
        return x

    def decode(self, x):
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, "postfilter"):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x
