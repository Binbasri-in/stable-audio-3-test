import typing as tp
import numpy as np
import torch
from torch.nn.functional import interpolate

from .sampling import sample_diffusion
from .utils import prepare_audio
from ..verbose import vprint


def generate_diffusion_uncond(
    model,
    steps: int = 250,
    batch_size: int = 1,
    sample_size: int = 2097152,
    seed: int = -1,
    device: str = "cuda",
    init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
    init_noise_level: float = 1.0,
    return_latents: bool = False,
    conditioning: tp.Optional[list] = None,
    conditioning_tensors: tp.Optional[dict] = None,
    adapt_duration_to_conditioning: tp.Optional[bool] = None,
    duration_padding_sec: float = 6.0,
    use_effective_length_for_schedule: tp.Optional[bool] = None,
    mask_padding_attention: tp.Optional[bool] = None,
    dist_shift=None,
    **sampler_kwargs,
) -> torch.Tensor:
    """Generate audio without Classifier-Free Guidance (CFG), requiring only a single pass of the model per step.

    Optionally accepts conditioning (e.g. duration metadata or empty prompt), but performs unconditional
    sampling (cfg_scale=1.0) with zero negative prompt evaluation.
    """
    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, "mask_padding_attention", False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, "use_effective_length_for_schedule", False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    audio_sample_size = sample_size

    if adapt_duration_to_conditioning and conditioning is not None:
        max_seconds = max((cond_dict.get("seconds_total", 0.0) for cond_dict in conditioning), default=0.0)
        if max_seconds > 0:
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)
            if model.pretransform is not None:
                ds_ratio = model.pretransform.downsampling_ratio
                align = ds_ratio
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            audio_sample_size = min(target_audio_samples, sample_size)

    latent_sample_size = audio_sample_size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio

    seed = seed if seed != -1 else int(np.random.randint(0, 2**32 - 1))
    vprint(f"seed: {seed}")
    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, latent_sample_size], device=device)

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)

    if conditioning is not None or conditioning_tensors is not None:
        if conditioning_tensors is None:
            conditioning_tensors = model.conditioner(conditioning, device)
        conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
        conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}
    else:
        conditioning_inputs = {}

    if init_audio is not None:
        in_sr, init_audio = init_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            if init_audio.shape[-1] > latent_sample_size:
                init_audio = init_audio[:, :, :latent_sample_size]
        init_audio = init_audio.repeat(batch_size, 1, 1)

    sampler_type = sampler_kwargs.pop("sampler_type", "pingpong")
    sampler_kwargs.pop("cfg_scale", None)

    return sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=conditioning_inputs,
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=1.0,
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        dist_shift=dist_shift if dist_shift is not None else getattr(model, "sampling_dist_shift", None),
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=1.0,
        init_data=init_audio,
        init_noise_level=init_noise_level,
        decode=not return_latents,
        **sampler_kwargs,
    ) 


def generate_diffusion_cond(
    model,
    steps: int = 250,
    cfg_scale: float = 6.0,
    conditioning: tp.Optional[list] = None,
    conditioning_tensors: tp.Optional[dict] = None,
    negative_conditioning: tp.Optional[list] = None,
    negative_conditioning_tensors: tp.Optional[dict] = None,
    batch_size: int = 1,
    sample_size: int = 2097152,
    seed: int = -1,
    device: str = "cuda",
    init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
    init_noise_level: float = 1.0,
    return_latents: bool = False,
    adapt_duration_to_conditioning: tp.Optional[bool] = None,
    duration_padding_sec: float = 6.0,
    use_effective_length_for_schedule: tp.Optional[bool] = None,
    mask_padding_attention: tp.Optional[bool] = None,
    apg_scale: float = 1.0,
    dist_shift=None,
    **sampler_kwargs,
) -> torch.Tensor:
    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, "mask_padding_attention", False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, "use_effective_length_for_schedule", False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    audio_sample_size = sample_size

    if adapt_duration_to_conditioning and conditioning is not None:
        max_seconds = max((cond_dict.get("seconds_total", 0.0) for cond_dict in conditioning), default=0.0)
        if max_seconds > 0:
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)
            if model.pretransform is not None:
                ds_ratio = model.pretransform.downsampling_ratio
                align = ds_ratio
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            audio_sample_size = min(target_audio_samples, sample_size)

    latent_sample_size = audio_sample_size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio

    seed = seed if seed != -1 else int(np.random.randint(0, 2**32 - 1))
    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, latent_sample_size], device=device)

    assert conditioning is not None or conditioning_tensors is not None, "Must provide conditioning"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    if init_audio is not None:
        in_sr, init_audio = init_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            if init_audio.shape[-1] > sample_size:
                init_audio = init_audio[:, :, :sample_size]
        init_audio = init_audio.repeat(batch_size, 1, 1)

    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}
    sampler_type = sampler_kwargs.pop("sampler_type", "pingpong")

    return sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        init_data=init_audio,
        init_noise_level=init_noise_level,
        decode=not return_latents,
        **sampler_kwargs,
    )


def generate_diffusion_cond_inpaint(
    model,
    steps: int = 250,
    cfg_scale: float = 6.0,
    conditioning: tp.Optional[list] = None,
    conditioning_tensors: tp.Optional[dict] = None,
    negative_conditioning: tp.Optional[list] = None,
    negative_conditioning_tensors: tp.Optional[dict] = None,
    batch_size: int = 1,
    sample_size: int = 2097152,
    seed: int = -1,
    device: str = "cuda",
    init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
    init_noise_level: float = 1.0,
    inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
    inpaint_mask=None,
    inpaint_mask_start_seconds: tp.Optional[float] = None,
    inpaint_mask_end_seconds: tp.Optional[float] = None,
    return_latents: bool = False,
    adapt_duration_to_conditioning: tp.Optional[bool] = None,
    duration_padding_sec: float = 6.0,
    use_effective_length_for_schedule: tp.Optional[bool] = None,
    mask_padding_attention: tp.Optional[bool] = None,
    apg_scale: float = 1.0,
    dist_shift=None,
    **sampler_kwargs,
) -> torch.Tensor:
    """Generate audio from a prompt using a diffusion inpainting model."""
    if mask_padding_attention is None:
        mask_padding_attention = getattr(model, "mask_padding_attention", False)
    if use_effective_length_for_schedule is None:
        use_effective_length_for_schedule = getattr(model, "use_effective_length_for_schedule", False)
    if adapt_duration_to_conditioning is None:
        adapt_duration_to_conditioning = mask_padding_attention or use_effective_length_for_schedule

    audio_sample_size = sample_size

    if adapt_duration_to_conditioning and conditioning is not None:
        max_seconds = max((cond_dict.get("seconds_total", 0.0) for cond_dict in conditioning), default=0.0)
        if max_seconds > 0:
            target_audio_samples = int((max_seconds + duration_padding_sec) * model.sample_rate)
            if model.pretransform is not None:
                ds_ratio = model.pretransform.downsampling_ratio
                align = ds_ratio
                target_audio_samples = ((target_audio_samples + align - 1) // align) * align
            audio_sample_size = min(target_audio_samples, sample_size)

    latent_sample_size = audio_sample_size
    if model.pretransform is not None:
        latent_sample_size = audio_sample_size // model.pretransform.downsampling_ratio

    sample_size = latent_sample_size

    if inpaint_mask_start_seconds is not None and inpaint_mask_end_seconds is not None:
        mask_start_samples = min(int(inpaint_mask_start_seconds * model.sample_rate), audio_sample_size)
        mask_end_samples = min(int(inpaint_mask_end_seconds * model.sample_rate), audio_sample_size)
        inpaint_mask = torch.ones(1, audio_sample_size, device=device)
        inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

    if inpaint_mask is not None and inpaint_mask.shape[-1] > audio_sample_size:
        inpaint_mask = inpaint_mask[:, :audio_sample_size]

    if inpaint_mask is not None and mask_padding_attention and conditioning is not None:
        max_seconds = max((c.get("seconds_total", 0.0) for c in conditioning), default=0.0)
        if max_seconds > 0:
            effective_audio_len = int(max_seconds * model.sample_rate)
            if effective_audio_len < inpaint_mask.shape[-1]:
                inpaint_mask = inpaint_mask.clone()
                inpaint_mask[:, effective_audio_len:] = 0

    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.float()

    seed = seed if seed != -1 else int(np.random.randint(0, 2**32 - 1))
    torch.manual_seed(seed)
    noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    assert conditioning is not None or conditioning_tensors is not None, "Must provide conditioning"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    if negative_conditioning is not None or negative_conditioning_tensors is not None:
        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        in_sr, init_audio = init_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)
            if init_audio.shape[-1] > sample_size:
                init_audio = init_audio[:, :, :sample_size]
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode="nearest").squeeze(1)
        init_audio = init_audio.repeat(batch_size, 1, 1)

    if inpaint_audio is not None:
        inpaint_sr, inpaint_audio = inpaint_audio
        io_channels = model.pretransform.io_channels if model.pretransform is not None else model.io_channels
        inpaint_audio = prepare_audio(inpaint_audio, in_sr=inpaint_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)
        if model.pretransform is not None:
            inpaint_audio = model.pretransform.encode(inpaint_audio)
            if inpaint_audio.shape[-1] > sample_size:
                inpaint_audio = inpaint_audio[:, :, :sample_size]
            if inpaint_mask is not None:
                inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode="nearest").squeeze(1)
        inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
    else:
        if inpaint_mask is not None:
            inpaint_mask = interpolate(inpaint_mask.unsqueeze(1), size=sample_size, mode="nearest").squeeze(1)

    if inpaint_mask is None:
        mask = torch.zeros((batch_size, 1, sample_size), device=device)
    else:
        mask = inpaint_mask.unsqueeze(1).to(device)

    if inpaint_audio is not None:
        inpaint_input = inpaint_audio * mask.expand_as(inpaint_audio)
    else:
        inpaint_input = torch.zeros((batch_size, model.io_channels, sample_size), device=device)

    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors["inpaint_mask"] = [mask]
        negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}
    sampler_type = sampler_kwargs.pop("sampler_type", "pingpong")

    return sample_diffusion(
        model=model.model,
        noise=noise,
        cond_inputs=cond_inputs,
        diffusion_objective=model.diffusion_objective,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_rate=model.sample_rate,
        pretransform=model.pretransform,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        headroom_seconds=duration_padding_sec,
        dist_shift=dist_shift if dist_shift is not None else model.sampling_dist_shift,
        sampler_type=sampler_type,
        batch_cfg=True,
        rescale_cfg=True,
        apg_scale=apg_scale,
        init_data=init_audio,
        init_noise_level=init_noise_level,
        decode=not return_latents,
        **sampler_kwargs,
    )
