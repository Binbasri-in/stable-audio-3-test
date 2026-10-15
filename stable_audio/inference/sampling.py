import math
import typing as tp
import torch
from tqdm import tqdm, trange

from .utils import compute_effective_seq_len_from_conditioning, create_padding_mask_from_lengths


def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift=None,
    effective_seq_len: tp.Union[int, torch.Tensor, None] = None,
    fallback_seq_len: tp.Optional[int] = None,
    include_endpoint: bool = True,
    device: tp.Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling."""
    n_points = steps + 1 if include_endpoint else steps
    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                t[..., 0] = sigma_max_tensor

    return t


def sample_discrete_euler(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Euler method."""
    t = sigmas.to(x.device)
    per_element_schedule = t.dim() == 2
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element_schedule:
            t_curr_tensor = t[:, i].to(x.dtype)
            t_prev = t[:, i + 1].to(x.dtype)
            dt = t_prev - t_curr_tensor
            dt_broadcast = dt.view(-1, 1, 1)
        else:
            t_curr = t[i]
            t_prev = t[i + 1]
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            dt = t_prev - t_curr
            dt_broadcast = dt

        v = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr_tensor[:, None, None] * v
            callback({"x": x, "t": t_curr_tensor, "sigma": t_curr_tensor, "i": i, "denoised": denoised})

        x = x + dt_broadcast * v

    return x


def sample_rk4(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. 4th-order Runge-Kutta."""
    ts = x.new_ones([x.shape[0]])
    t = sigmas.to(x.device)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr

        k1 = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr * k1
            callback({"x": x, "t": t_curr, "sigma": t_curr, "i": i, "denoised": denoised})

        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)

        t_prev_eval = t_prev.clamp(min=1e-5)
        k4 = model(x + dt * k3, t_prev_eval * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def sample_flow_dpmpp(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. DPM-Solver++ for RF models."""
    t = sigmas.to(x.device)
    per_element_schedule = t.dim() == 2
    num_steps = t.shape[-1] - 1
    old_denoised = None

    log_snr = lambda t_: ((1 - t_).clamp(min=1e-10) / t_.clamp(min=1e-10)).log()

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            t_curr = t[:, i]
            t_next = t[:, i + 1]
            t_prev = t[:, i - 1] if i > 0 else None
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr
        else:
            t_curr = t[i]
            t_next = t[i + 1]
            t_prev = t[i - 1] if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr.expand(x.shape[0])

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        if callback is not None:
            callback({"x": x, "i": i, "t": t_curr, "sigma": t_curr, "denoised": denoised})

        alpha_t = 1 - t_next_broadcast
        dt = t_next_broadcast - t_curr_broadcast
        dpmpp_coeff = dt / ((1 - t_next_broadcast).clamp(min=1e-10) * t_curr_broadcast.clamp(min=1e-10))

        is_first_step = old_denoised is None
        is_last_step = (t_next_broadcast == 0).all() if per_element_schedule else (t_next == 0)

        if is_first_step or is_last_step:
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised
        else:
            t_prev_broadcast = t_prev.view(-1, 1, 1) if per_element_schedule else t_prev
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised
    return x


def sample_flow_pingpong(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
    """Draws samples from a model given starting noise. Ping-pong sampling for rectified flow."""
    t = sigmas.to(x.device)
    per_element_schedule = t.dim() == 2
    num_steps = t.shape[-1] - 1

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            t_curr = t[:, i].to(x.dtype)
            t_next = t[:, i + 1].to(x.dtype)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr
        else:
            t_curr = t[i].to(x.dtype)
            t_next = t[i + 1].to(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "t": t_curr, "sigma": t_curr, "sigma_hat": t_curr, "denoised": denoised})

        x = (1 - t_next_broadcast) * denoised + t_next_broadcast * torch.randn_like(x)

    return x


@torch.no_grad()
def sample_diffusion(
    model,
    noise: torch.Tensor,
    cond_inputs: dict,
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    conditioning: tp.Optional[tp.List[dict]] = None,
    sample_rate: int = 44100,
    pretransform=None,
    mask_padding_attention: bool = False,
    use_effective_length_for_schedule: bool = False,
    headroom_seconds: float = 5.0,
    padding_mask: tp.Optional[torch.Tensor] = None,
    dist_shift=None,
    sampler_type: str = "pingpong",
    batch_cfg: bool = True,
    rescale_cfg: bool = False,
    apg_scale: float = 1.0,
    init_data: tp.Optional[torch.Tensor] = None,
    init_noise_level: float = 1.0,
    callback=None,
    disable_tqdm: bool = False,
    decode: bool = True,
    **sampler_kwargs,
) -> torch.Tensor:
    device = noise.device
    latent_seq_len = noise.shape[-1]
    downsampling_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

    if sampler_type is None:
        sampler_type = "pingpong" if diffusion_objective == "rf_denoiser" else "euler"

    effective_seq_len = None
    if use_effective_length_for_schedule and conditioning is not None:
        effective_seq_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )

    if padding_mask is None and mask_padding_attention and conditioning is not None:
        raw_effective_len = compute_effective_seq_len_from_conditioning(
            conditioning, sample_rate, downsampling_ratio, device
        )
        if raw_effective_len is not None:
            headroom_tokens = int(headroom_seconds * sample_rate / downsampling_ratio)
            valid_lengths = (raw_effective_len + headroom_tokens).clamp(max=latent_seq_len).long()
            padding_mask = create_padding_mask_from_lengths(valid_lengths, latent_seq_len)

    sigma_max = init_noise_level if init_data is not None else sampler_kwargs.get("sigma_max", 1.0)

    if init_data is not None and diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        noise = init_data * (1 - sigma_max) + noise * sigma_max

    common_kwargs = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        "batch_cfg": batch_cfg,
        "rescale_cfg": rescale_cfg,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
        **sampler_kwargs,
    }
    common_kwargs.pop("sigma_min", None)
    common_kwargs.pop("sigma_max", None)
    common_kwargs.pop("rho", None)

    sigmas = build_schedule(
        steps=steps,
        sigma_max=sigma_max,
        dist_shift=dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=latent_seq_len,
        include_endpoint=True,
        device=device,
    )

    if sampler_type == "euler":
        sampled = sample_discrete_euler(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
    elif sampler_type == "rk4":
        sampled = sample_rk4(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
    elif sampler_type == "dpmpp":
        sampled = sample_flow_dpmpp(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
    elif sampler_type == "pingpong":
        sampled = sample_flow_pingpong(model, noise, sigmas=sigmas, callback=callback, disable_tqdm=disable_tqdm, **common_kwargs)
    else:
        raise ValueError(f"Unknown sampler_type for {diffusion_objective}: {sampler_type}")

    if decode and pretransform is not None:
        sampled = sampled.to(next(pretransform.parameters()).dtype)
        sampled = pretransform.decode(sampled)

        if padding_mask is not None:
            audio_mask = padding_mask.unsqueeze(1).repeat_interleave(downsampling_ratio, dim=-1)
            if audio_mask.shape[-1] > sampled.shape[-1]:
                audio_mask = audio_mask[..., : sampled.shape[-1]]
            elif audio_mask.shape[-1] < sampled.shape[-1]:
                audio_mask = torch.nn.functional.pad(audio_mask, (0, sampled.shape[-1] - audio_mask.shape[-1]), value=False)
            sampled = sampled * audio_mask.to(sampled.dtype)

    return sampled
