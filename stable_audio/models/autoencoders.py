import math
from typing import Any, Dict, List, Literal, Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms as T

from .attn_masks import generate_chunked_sliding_window_mask, generate_sliding_window_mask
from .blocks import Transpose, WNConv1d, WNConvTranspose1d, _zero_pad_modulo_sequence, checkpoint
from .bottleneck import Bottleneck, DiscreteBottleneck
from .pretransforms import Pretransform
from .transformer import TransformerBlock


class TransformerResamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        sliding_window=None,
        chunk_size=128,
        chunk_midpoint_shift=False,
        type="encoder",
        transformer_depth=3,
        checkpointing=False,
        conformer=False,
        layer_scale=False,
        dim_heads=128,
        differential=True,
        variable_stride=False,
        feat_scale=False,
        sinusoidal_blocks=0,
        mask_noise=0,
        ff_mult=3,
        mapping_bias=True,
        cross_attn=False,
        dyt=True,
        conv_mapping=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        if type not in ["encoder", "decoder"]:
            raise ValueError(f"Unknown type {type}. Must be 'encoder' or 'decoder'")

        self.checkpointing = checkpointing
        transformer_dim = out_channels if type == "encoder" else in_channels
        transformers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.variable_stride = variable_stride
        self.stride = stride
        self.mapping = (
            WNConv1d(in_channels, out_channels, 3 if conv_mapping else 1, padding="same", bias=mapping_bias)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.chunk_size = chunk_size
        self.chunk_midpoint_shift = chunk_midpoint_shift
        self.type = type
        self.mask_noise = mask_noise
        self.sliding_window_latents = sliding_window

        self.sliding_window_seq = self._get_sliding_window_size(sliding_window, stride)
        self.input_seg_size, self.output_seg_size, self.sub_chunk_size = self._get_seg_sizes(stride)
        self.transformer_depth = transformer_depth
        for i in range(transformer_depth):
            sinusoidal = True if ((transformer_depth - i) < sinusoidal_blocks) else False
            transformers.append(
                TransformerBlock(
                    transformer_dim,
                    dim_heads=dim_heads,
                    causal=False,
                    zero_init_branch_outputs=True if not layer_scale else False,
                    norm_type="dyt" if dyt else "rms_norm",
                    conformer=conformer,
                    layer_scale=layer_scale,
                    add_rope=True,
                    attn_kwargs={
                        "qk_norm": "dyt" if dyt else "rms",
                        "qk_norm_eps": 1e-3,
                        "differential": differential,
                        "feat_scale": feat_scale,
                    },
                    ff_kwargs={"mult": ff_mult, "no_bias": False, "sinusoidal": sinusoidal},
                    norm_kwargs={"eps": 1e-3},
                    cross_attend=cross_attn,
                )
            )

        self.new_tokens = nn.Parameter(
            1e-5 * torch.randn(1, self.output_seg_size if not self.variable_stride else 1, out_channels if type == "encoder" else in_channels)
        )

        self.transformers = nn.ModuleList(transformers)

        if freeze_backbone:
            for param in self.transformers.parameters():
                param.requires_grad = False
            self.new_tokens.requires_grad = False

    def _get_sliding_window_size(self, window, stride, prepend_cond_length=0):
        if window is None:
            return None
        return [(win * (stride + 1 + prepend_cond_length)) for win in window]

    def _get_seg_sizes(self, stride, prepend_cond_length=0):
        sub_chunk_size = stride + 1 + prepend_cond_length
        if self.sliding_window_latents is None:
            assert (self.chunk_size % stride) == 0, f"Stride must fit evenly into chunk size:{self.chunk_size}"
        input_seg_size = stride if self.type == "encoder" else 1
        output_seg_size = 1 if self.type == "encoder" else stride
        return input_seg_size, output_seg_size, sub_chunk_size

    def forward(
        self,
        x,
        stride=None,
        return_features=False,
        override_new_tokens=None,
        prepend_cond=None,
        cross_attn_cond=None,
    ):
        batch_size = x.shape[0]
        if return_features:
            features = []

        if stride is None:
            input_seg_size = self.input_seg_size
            output_seg_size = self.output_seg_size
            sub_chunk_size = self.sub_chunk_size
            sliding_window = self.sliding_window_seq
        else:
            prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
            input_seg_size, output_seg_size, sub_chunk_size = self._get_seg_sizes(stride, prepend_cond_length)
            sliding_window = self._get_sliding_window_size(self.sliding_window_latents, stride, prepend_cond_length)

        if self.type == "encoder":
            if self.transformer_depth > 0:
                pad_modulo = self.chunk_size if sliding_window is None else input_seg_size
                x = _zero_pad_modulo_sequence(x, pad_modulo, dim=-1)
            x = self.mapping(x)

        if self.transformer_depth > 0:
            x = rearrange(x, "... a b -> ... b a")
            if return_features:
                features.append(x)
            if self.type != "encoder":
                if sliding_window is None:
                    active_stride = stride if stride is not None else self.stride
                    pad_modulo = self.chunk_size // active_stride
                    x = _zero_pad_modulo_sequence(x, pad_modulo)
                else:
                    x = _zero_pad_modulo_sequence(x, input_seg_size)
            x = rearrange(x, "b (n c) d -> (b n) c d", c=input_seg_size)
            new_token_seq_dim = -1 if not self.variable_stride else output_seg_size
            new_tokens = self.new_tokens.expand([x.shape[0], new_token_seq_dim, -1])
            if override_new_tokens is not None:
                override_new_tokens = rearrange(override_new_tokens, "b (n c) d -> (b n) c d", c=output_seg_size)
                new_tokens = new_tokens + override_new_tokens
            elif self.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * self.mask_noise
            x = torch.cat([x, new_tokens], dim=-2)
            if prepend_cond is not None:
                n = x.shape[0] // batch_size
                cond_folded = (
                    prepend_cond.unsqueeze(1)
                    .expand(batch_size, n, prepend_cond.shape[-2], x.shape[-1])
                    .reshape(n * batch_size, prepend_cond.shape[-2], x.shape[-1])
                )
                x = torch.cat([cond_folded, x], dim=-2)
            x = rearrange(x, "(b n) c d -> b (n c) d", b=batch_size)

            if sliding_window is None:
                prepend_cond_length = prepend_cond.shape[-2] if prepend_cond is not None else 0
                effective_chunk_size = (
                    self.chunk_size
                    + self.chunk_size * (1 + prepend_cond_length) // (stride if stride is not None else self.stride)
                )

            if sliding_window is None and self.chunk_midpoint_shift:
                split = self.transformer_depth // 2
                shift = effective_chunk_size // 2

                nc = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_first = None
                if cross_attn_cond is not None:
                    cross_attn_first = cross_attn_cond.repeat_interleave(nc, dim=0)
                for layer in self.transformers[:split]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context=cross_attn_first, self_attention_flash_sliding_window=None)
                    else:
                        x = layer(x, context=cross_attn_first)
                    if return_features:
                        features.append(rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size))
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

                x = torch.cat([x[:, :shift, :], x, x[:, -shift:, :]], dim=1)
                nc_shifted = x.shape[1] // effective_chunk_size
                x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)
                cross_attn_second = None
                if cross_attn_cond is not None:
                    cross_attn_second = cross_attn_cond.repeat_interleave(nc_shifted, dim=0)
                for layer in self.transformers[split:]:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context=cross_attn_second, self_attention_flash_sliding_window=None)
                    else:
                        x = layer(x, context=cross_attn_second)
                    if return_features:
                        feat = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
                        features.append(feat[:, shift:-shift, :])
                x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)
                x = x[:, shift:-shift, :]
            else:
                if sliding_window is None:
                    x = rearrange(x, "b (nc cc) d -> (b nc) cc d", cc=effective_chunk_size)

                for layer in self.transformers:
                    if self.checkpointing:
                        x = checkpoint(layer, x, context=cross_attn_cond, self_attention_flash_sliding_window=sliding_window)
                    else:
                        x = layer(x, context=cross_attn_cond, self_attention_flash_sliding_window=sliding_window)
                    if return_features:
                        features.append(x)

                if sliding_window is None:
                    x = rearrange(x, "(b nc) cc d -> b (nc cc) d", b=batch_size)

            x = rearrange(x, "b (n c) d -> (b n) c d", c=sub_chunk_size)
            x = x[:, -output_seg_size:, :]
            x = rearrange(x, "(b n) c d -> b d (n c)", b=batch_size)
        if self.type == "decoder":
            x = self.mapping(x)
        if return_features:
            return x, features
        return x


class SAMEEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        transformer_depths=[3, 3, 3, 3],
        sliding_window=None,
        checkpointing=False,
        conformer=False,
        layer_scale=False,
        causal=False,
        differential=True,
        variable_stride=False,
        mask_noise=0.0,
        conv_mapping=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.strides = strides

        channel_dims = [c * channels for c in c_mults]
        channel_dims = [in_channels] + channel_dims

        self.depth = len(c_mults)
        layers = []

        for i in range(self.depth):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i + 1],
                    stride=strides[i],
                    transformer_depth=transformer_depths[i],
                    sliding_window=sliding_window,
                    checkpointing=checkpointing,
                    conformer=conformer,
                    layer_scale=layer_scale,
                    causal=causal,
                    differential=differential,
                    variable_stride=variable_stride,
                    mask_noise=mask_noise,
                    conv_mapping=conv_mapping,
                    freeze_backbone=freeze_backbone,
                    **kwargs,
                )
            )

        layers += [Transpose(), nn.Linear(channel_dims[-1], latent_dim), Transpose()]
        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[-2].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride=None, return_features=False, **kwargs):
        if override_stride is not None:
            assert isinstance(override_stride, list) and len(override_stride) == self.depth
        features = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerResamplingBlock):
                stride = override_stride[i] if override_stride is not None else None
                if return_features:
                    x, feats = layer(x, stride=stride, return_features=True)
                    features.extend(feats)
                else:
                    x = layer(x, stride=stride)
            else:
                x = layer(x)
        if return_features:
            return x, features
        return x


class SAMEDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        transformer_depths=[3, 3, 3, 3],
        sliding_window=None,
        checkpointing=False,
        conformer=False,
        layer_scale=False,
        causal=False,
        differential=True,
        variable_stride=False,
        sinusoidal_blocks=[0, 0, 0, 0],
        mask_noise=0.0,
        conv_mapping=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        channel_dims = [c * channels for c in c_mults]
        channel_dims = [out_channels] + channel_dims

        self.depth = len(c_mults)
        layers = [Transpose(), nn.Linear(latent_dim, channel_dims[-1]), Transpose()]

        for i in range(self.depth, 0, -1):
            layers.append(
                TransformerResamplingBlock(
                    in_channels=channel_dims[i],
                    out_channels=channel_dims[i - 1],
                    stride=strides[i - 1],
                    type="decoder",
                    transformer_depth=transformer_depths[i - 1],
                    sliding_window=sliding_window,
                    checkpointing=checkpointing,
                    conformer=conformer,
                    layer_scale=layer_scale,
                    causal=causal,
                    differential=differential,
                    variable_stride=variable_stride,
                    sinusoidal_blocks=sinusoidal_blocks[i - 1] if i - 1 < len(sinusoidal_blocks) else 0,
                    mask_noise=mask_noise,
                    conv_mapping=conv_mapping,
                    freeze_backbone=freeze_backbone,
                    **kwargs,
                )
            )

        self.layers = nn.ModuleList(layers)

        if freeze_backbone:
            for param in self.layers[1].parameters():
                param.requires_grad = False

    def forward(self, x, override_stride=None, **kwargs):
        if override_stride is not None:
            assert isinstance(override_stride, list) and len(override_stride) == self.depth
        transformer_layer_index = 0
        for layer in self.layers:
            if isinstance(layer, TransformerResamplingBlock):
                stride = override_stride[transformer_layer_index] if override_stride is not None else None
                x = layer(x, stride=stride)
                transformer_layer_index += 1
            else:
                x = layer(x)
        return x


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels=None,
        out_channels=None,
        soft_clip=False,
        freeze_pretransform=False,
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels or io_channels
        self.out_channels = out_channels or io_channels
        self.min_length = self.downsampling_ratio
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.pretransform = pretransform
        self.freeze_pretransform = freeze_pretransform

        if self.pretransform is not None:
            for p in self.pretransform.parameters():
                p.requires_grad = not self.freeze_pretransform

        self.soft_clip = soft_clip
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, return_pretransform=False, **kwargs):
        info = {}
        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    audio = self.pretransform.encode(audio)

        latents = self.encoder(audio, **kwargs) if self.encoder is not None else audio
        if self.bottleneck is not None:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)

        if return_info and return_pretransform:
            return latents, info, audio
        elif return_info:
            return latents, info
        elif return_pretransform:
            return latents, audio
        return latents

    def decode(self, latents, iterate_batch=False, return_loss=False, **kwargs):
        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)

        return decoded

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        samples_per_latent = int(self.downsampling_ratio)
        if not chunked or audio.shape[-1] < chunk_size * samples_per_latent:
            return self.encode(audio, **kwargs)

        chunk_size_samples = chunk_size * samples_per_latent
        hop_samples = (chunk_size - overlap) * samples_per_latent
        total_samples = audio.shape[-1]

        chunk_starts = list(range(0, total_samples - chunk_size_samples + 1, hop_samples))
        if chunk_starts[-1] != total_samples - chunk_size_samples:
            chunk_starts.append(total_samples - chunk_size_samples)

        encoded_chunks = [self.encode(audio[..., s : s + chunk_size_samples]) for s in chunk_starts]

        total_latents = total_samples // samples_per_latent
        half_overlap_latents = overlap // 2
        output = audio.new_zeros(*encoded_chunks[0].shape[:-1], total_latents)
        num_chunks = len(chunk_starts)

        for i, (start_sample, chunk) in enumerate(zip(chunk_starts, encoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_latents - chunk_size) if is_last else (start_sample // samples_per_latent)
            left = 0 if is_first else half_overlap_latents
            right = chunk_size if is_last else chunk_size - half_overlap_latents
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output

    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        if not chunked or latents.shape[-1] < chunk_size:
            return self.decode(latents, **kwargs)

        samples_per_latent = int(self.downsampling_ratio)
        hop_latents = chunk_size - overlap
        total_latents = latents.shape[-1]

        chunk_starts = list(range(0, total_latents - chunk_size + 1, hop_latents))
        if chunk_starts[-1] != total_latents - chunk_size:
            chunk_starts.append(total_latents - chunk_size)

        decoded_chunks = [self.decode(latents[..., s : s + chunk_size]) for s in chunk_starts]

        total_samples = total_latents * samples_per_latent
        chunk_size_samples = chunk_size * samples_per_latent
        half_overlap_samples = (overlap // 2) * samples_per_latent

        output = latents.new_zeros(*decoded_chunks[0].shape[:-1], total_samples)
        num_chunks = len(chunk_starts)

        for i, (start_latent, chunk) in enumerate(zip(chunk_starts, decoded_chunks)):
            is_first = i == 0
            is_last = i == num_chunks - 1
            out_start = (total_samples - chunk_size_samples) if is_last else (start_latent * samples_per_latent)
            left = 0 if is_first else half_overlap_samples
            right = chunk_size_samples if is_last else chunk_size_samples - half_overlap_samples
            output[..., out_start + left : out_start + right] = chunk[..., left:right]

        return output


def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    if encoder_type in ("same", "taae_v2"):
        encoder = SAMEEncoder(**encoder_config["config"])
    else:
        raise ValueError(f"Unsupported encoder type for Stable Audio 3: {encoder_type}")

    if not encoder_config.get("requires_grad", True):
        for param in encoder.parameters():
            param.requires_grad = False
    return encoder


def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    if decoder_type in ("same", "taae_v2"):
        decoder = SAMEDecoder(**decoder_config["config"])
    else:
        raise ValueError(f"Unsupported decoder type for Stable Audio 3: {decoder_type}")

    if not decoder_config.get("requires_grad", True):
        for param in decoder.parameters():
            param.requires_grad = False
    return decoder
