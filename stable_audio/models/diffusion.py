import math
import typing as tp
import numpy as np
import torch
from torch import nn

from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .pretransforms import Pretransform


class IdentityDistributionShift:
    """No-op distribution shift — returns timesteps unchanged."""

    def shift(self, t: torch.Tensor, seq_len):
        return t


class FluxDistributionShift:
    def __init__(self, min_length=256, max_length=4096, alpha_min=1.0, alpha_max=1.0):
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            self.log_max_seq += 1e-8

    def get_alpha(self, seq_len: tp.Union[int, torch.Tensor]):
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.float().clamp(self.min_length, self.max_length)
            log_seq = torch.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return torch.exp(log_alpha)
        else:
            seq_len = max(min(seq_len, self.max_length), self.min_length)
            log_seq = math.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return math.exp(log_alpha)

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        alpha = self.get_alpha(seq_len)
        if isinstance(seq_len, torch.Tensor):
            alpha = alpha.to(t.device)
            if t.dim() == 1 and alpha.dim() == 1 and t.shape[0] != alpha.shape[0]:
                t = t.unsqueeze(0)
                alpha = alpha.unsqueeze(1)
        return alpha * t / (1 + (alpha - 1.0) * t)


class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.to(t.device)
            seq_len_clamped = seq_len.float().clamp(self.min_length, self.max_length)
            if t.dim() == 1 and seq_len_clamped.dim() == 1 and t.shape[0] != seq_len_clamped.shape[0]:
                t = t.unsqueeze(0)
                seq_len_clamped = seq_len_clamped.unsqueeze(1)
            sigma = 1.0
            mu = -(self.base_shift + (self.max_shift - self.base_shift) * (seq_len_clamped - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - torch.exp(mu) / (torch.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        else:
            seq_len = min(max(seq_len, self.min_length), self.max_length)
            sigma = 1.0
            mu = -(self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
            t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        return t_out


class LogSNRShift:
    def __init__(self, anchor_length=2000, anchor_logsnr=-6.2, rate=1.0, logsnr_end=2.0):
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len):
        if isinstance(seq_len, torch.Tensor):
            log2_ratio = torch.log2(seq_len.float() / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio
        else:
            log2_ratio = math.log2(seq_len / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio

    def shift(self, t: torch.Tensor, seq_len: tp.Union[int, torch.Tensor]):
        t_original = t
        logsnr_start = self.get_logsnr_start(seq_len)
        if isinstance(seq_len, torch.Tensor):
            logsnr_start = logsnr_start.to(t.device)
            if t.dim() == 1 and logsnr_start.dim() == 1 and t.shape[0] != logsnr_start.shape[0]:
                t = t.unsqueeze(0)
                logsnr_start = logsnr_start.unsqueeze(1)
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)
        return t_out


class ConditionedDiffusionModel(nn.Module):
    def __init__(
        self,
        *args,
        supports_cross_attention: bool = False,
        supports_input_concat: bool = False,
        supports_global_cond: bool = False,
        supports_prepend_cond: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class ConditionedDiffusionModelWrapper(nn.Module):
    def __init__(
        self,
        model: ConditionedDiffusionModel,
        conditioner: MultiConditioner,
        io_channels: int,
        sample_rate: int,
        min_input_length: int,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "rf_denoiser",
        distribution_shift_options=None,
        sampling_distribution_shift_options=None,
        mask_padding_attention: bool = False,
        use_effective_length_for_schedule: bool = False,
        pretransform: tp.Optional[Pretransform] = None,
        cross_attn_cond_ids: tp.List[str] = [],
        global_cond_ids: tp.List[str] = [],
        input_concat_ids: tp.List[str] = [],
        local_add_cond_ids: tp.List[str] = [],
        modular_local_cond_ids: tp.List[str] = [],
        prepend_cond_ids: tp.List[str] = [],
    ):
        super().__init__()
        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.local_add_cond_ids = local_add_cond_ids
        self.modular_local_cond_ids = modular_local_cond_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length
        self.mask_padding_attention = mask_padding_attention
        self.use_effective_length_for_schedule = use_effective_length_for_schedule

        self.dist_shift = None
        if distribution_shift_options is not None:
            self.dist_shift = self._create_dist_shift(distribution_shift_options)

        if sampling_distribution_shift_options is not None:
            self.sampling_dist_shift = self._create_dist_shift(sampling_distribution_shift_options)
        else:
            self.sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)

    @staticmethod
    def _create_dist_shift(options: dict):
        dist_shift_type = options.get("type", "full")
        kwargs = {k: v for k, v in options.items() if k != "type"}
        if dist_shift_type == "none":
            return IdentityDistributionShift()
        elif dist_shift_type == "flux":
            return FluxDistributionShift(**kwargs)
        elif dist_shift_type == "full":
            return DistributionShift(**kwargs)
        elif dist_shift_type == "logsnr":
            return LogSNRShift(**kwargs)
        else:
            raise ValueError(f"Unknown distribution shift type: {dist_shift_type}")

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None
        local_add_cond = None
        modular_local_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            cross_attention_input = []
            cross_attention_masks = []
            for key in self.cross_attn_cond_ids:
                cross_attn_in, cross_attn_mask = conditioning_tensors[key]
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)
                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)
            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]
                global_conds.append(global_cond_input)
            global_cond = torch.cat(global_conds, dim=-1)
            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            input_concat_cond = torch.cat([conditioning_tensors[key][0] for key in self.input_concat_ids], dim=1)

        if len(self.local_add_cond_ids) > 0:
            add_tensors = [conditioning_tensors[key][0] for key in self.local_add_cond_ids if key in conditioning_tensors]
            local_add_cond = torch.cat(add_tensors, dim=1) if len(add_tensors) > 0 else None

        if len(self.modular_local_cond_ids) > 0:
            modular_local_cond = {}
            for key in self.modular_local_cond_ids:
                if key in conditioning_tensors:
                    modular_local_cond[key] = conditioning_tensors[key][0]
            if len(modular_local_cond) == 0:
                modular_local_cond = None

        if len(self.prepend_cond_ids) > 0:
            prepend_conds = []
            prepend_cond_masks = []
            for key in self.prepend_cond_ids:
                if key in conditioning_tensors:
                    prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                    prepend_conds.append(prepend_cond_input)
                    prepend_cond_masks.append(prepend_cond_mask)
            prepend_cond = torch.cat(prepend_conds, dim=1) if len(prepend_conds) > 0 else None
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1) if len(prepend_cond_masks) > 0 else None

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond,
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask,
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)


class DiTWrapper(ConditionedDiffusionModel):
    def __init__(self, diffusion_objective: str, *args, **kwargs):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)
        self.diffusion_objective = diffusion_objective
        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        local_add_cond=None,
        negative_input_concat_cond=None,
        global_cond=None,
        negative_global_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob: float = 0.0,
        batch_cfg: bool = True,
        rescale_cfg: bool = False,
        scale_phi: float = 0.0,
        **kwargs,
    ):
        assert batch_cfg or cfg_scale == 1.0, "batch_cfg must be True for DiTWrapper when cfg_scale != 1.0"
        global_embed = global_cond if global_cond is not None else global_embed
        negative_global_embed = negative_global_cond if negative_global_cond is not None else negative_global_embed
        kwargs.pop("global_cond", None)
        kwargs.pop("negative_global_cond", None)

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            input_concat_cond=input_concat_cond,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_embed,
            negative_global_embed=negative_global_embed,
            local_add_cond=local_add_cond,
            **kwargs,
        )


def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):
    from ..factory import create_pretransform_from_config

    model_config = config["model"]
    diffusion_config = model_config.get("diffusion", None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_objective = diffusion_config.get("diffusion_objective", "rf_denoiser")
    diffusion_model_type = diffusion_config.get("type", None)
    diffusion_model_config = diffusion_config.get("config", None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    modular_local_cond_configs = diffusion_config.get("modular_local_cond_configs", [])

    if diffusion_model_type == "dit":
        diffusion_model = DiTWrapper(
            diffusion_objective=diffusion_objective,
            modular_local_cond_configs=modular_local_cond_configs,
            **diffusion_model_config,
        )
    else:
        raise ValueError(f"Unsupported diffusion model type for Stable Audio 3: {diffusion_model_type}")

    io_channels = model_config.get("io_channels", None)
    sample_rate = config.get("sample_rate", None)
    assert io_channels is not None and sample_rate is not None

    cross_attention_ids = diffusion_config.get("cross_attention_cond_ids", [])
    global_cond_ids = diffusion_config.get("global_cond_ids", [])
    input_concat_ids = diffusion_config.get("input_concat_ids", [])
    local_add_cond_ids = diffusion_config.get("local_add_cond_ids", [])
    modular_local_cond_ids = [c["id"] for c in modular_local_cond_configs]
    prepend_cond_ids = diffusion_config.get("prepend_cond_ids", [])

    pretransform = model_config.get("pretransform", None)
    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    conditioning_config = model_config.get("conditioning", None)
    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config, pretransform=pretransform)

    min_input_length *= diffusion_model.model.patch_size

    return ConditionedDiffusionModelWrapper(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        local_add_cond_ids=local_add_cond_ids,
        modular_local_cond_ids=modular_local_cond_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        diffusion_objective=diffusion_objective,
        distribution_shift_options=diffusion_config.get("distribution_shift_options", None),
        sampling_distribution_shift_options=diffusion_config.get("sampling_distribution_shift_options", None),
        mask_padding_attention=diffusion_config.get("mask_padding_attention", False),
        use_effective_length_for_schedule=diffusion_config.get("use_effective_length_for_schedule", False),
    )
