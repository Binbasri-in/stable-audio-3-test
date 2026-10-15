import math
from typing import Optional, Tuple

import torch
from torch import nn
from torchaudio import transforms as T


class PadCrop(nn.Module):
    def __init__(self, n_samples: int, randomize: bool = False):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        n, s = signal.shape[-2], signal.shape[-1]
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([*signal.shape[:-1], self.n_samples])
        output[..., : min(s, self.n_samples)] = signal[..., start:end]
        return output


def set_audio_channels(audio: torch.Tensor, target_channels: int) -> torch.Tensor:
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)

    if target_channels == 1:
        audio = audio.mean(1, keepdim=True)
    elif target_channels == 2:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    return audio


def prepare_audio(
    audio: torch.Tensor,
    in_sr: int,
    target_sr: int,
    target_length: int,
    target_channels: int,
    device: torch.device | str,
) -> torch.Tensor:
    audio = audio.to(device)

    if in_sr != target_sr:
        resample_tf = T.Resample(in_sr, target_sr).to(device)
        audio = resample_tf(audio)

    audio = PadCrop(target_length, randomize=False)(audio)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = set_audio_channels(audio, target_channels)
    return audio


def create_padding_mask_from_lengths(
    valid_lengths: torch.Tensor,
    total_seq_len: int,
) -> torch.Tensor:
    """Create a boolean padding mask from per-batch valid sequence lengths."""
    device = valid_lengths.device
    positions = torch.arange(total_seq_len, device=device).unsqueeze(0)
    padding_mask = positions < valid_lengths.unsqueeze(1)
    return padding_mask


def compute_effective_seq_len_from_conditioning(
    conditioning: list,
    sample_rate: int,
    downsampling_ratio: int = 1,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """Compute effective sequence lengths from seconds_total in conditioning dicts."""
    if conditioning is None:
        return None

    if not any("seconds_total" in c for c in conditioning):
        return None

    effective_lengths = []
    for cond_dict in conditioning:
        if "seconds_total" in cond_dict:
            seconds = cond_dict["seconds_total"]
            audio_samples = int(seconds * sample_rate)
            latent_length = math.ceil(audio_samples / downsampling_ratio)
            effective_lengths.append(latent_length)
        else:
            effective_lengths.append(None)

    if any(l is None for l in effective_lengths):
        return None

    return torch.tensor(effective_lengths, dtype=torch.float32, device=device)
