"""Notebook visualizations for source separation samples.

The public helpers accept either a :class:`SeparationPairSample` returned by a
dataset or a batch dictionary returned by ``SourceSeparationCollator``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .common import SeparationPairSample

AudioLike = Union[Tensor, np.ndarray]
SampleLike = Union[SeparationPairSample, Mapping[str, Any]]


def _value(sample: SampleLike, name: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _sample_value(value: Any, sample_idx: int) -> Any:
    """Select one item from collated metadata or tensors, preserving scalars."""
    if value is None or isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Tensor) and value.ndim > 0:
        return value[sample_idx]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return value[sample_idx]
    if isinstance(value, (list, tuple)):
        return value[sample_idx]
    return value


def _waveform(audio: AudioLike, sample_idx: int = 0) -> np.ndarray:
    array = audio.detach().cpu().numpy() if isinstance(audio, Tensor) else np.asarray(audio)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[sample_idx]
    if array.ndim == 2:
        if array.shape[0] > array.shape[1] and array.shape[1] <= 8:
            array = array.T
        array = array.mean(axis=0)
    if array.ndim != 1:
        raise ValueError(f"Expected audio shaped [samples], [channels, samples], or [batch, channels, samples], got {array.shape}")
    return array


def _spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
) -> np.ndarray:
    """Compute a log-mel spectrogram without requiring a dataset instance."""
    transform = torch.stft
    waveform = torch.from_numpy(audio)
    if waveform.numel() < n_fft:
        waveform = torch.nn.functional.pad(waveform, (0, n_fft - waveform.numel()))
    window = torch.hann_window(n_fft)
    stft = transform(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    power = stft.abs().square()
    if n_mels != power.shape[0]:
        frequencies = torch.linspace(0, sample_rate / 2, power.shape[0])
        mel_points = torch.linspace(0, sample_rate / 2, n_mels + 2)
        weights = torch.zeros(n_mels, power.shape[0])
        for index in range(n_mels):
            left, center, right = mel_points[index : index + 3]
            weights[index] = torch.clamp(
                torch.minimum(
                    (frequencies - left) / max(float(center - left), 1e-8),
                    (right - frequencies) / max(float(right - center), 1e-8),
                ),
                min=0.0,
            )
        power = weights @ power
    return torch.log(power.clamp_min(1e-5)).numpy()


def _theme(dark: bool) -> Dict[str, str]:
    return (
        {"background": "#181825", "figure": "#11111b", "text": "#cdd6f4", "grid": "#45475a"}
        if dark
        else {"background": "#ffffff", "figure": "#ffffff", "text": "#1e1e2e", "grid": "#bcc0cc"}
    )


def plot_waveforms(
    waveforms: Mapping[str, AudioLike] | Sequence[AudioLike],
    sample_rate: int = 44100,
    titles: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dark_theme: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot one or more time-domain waveforms and return ``(figure, axes)``."""
    items = list(waveforms.items()) if isinstance(waveforms, Mapping) else list(enumerate(waveforms))
    if not items:
        raise ValueError("waveforms must contain at least one waveform")
    labels = list(titles) if titles is not None else [str(label) for label, _ in items]
    theme = _theme(dark_theme)
    fig, axes = plt.subplots(len(items), 1, squeeze=False, sharex=True,
                             figsize=figsize or (12, max(2.2 * len(items), 4)))
    axes = axes[:, 0]
    fig.patch.set_facecolor(theme["figure"])
    colors = ("#89b4fa", "#a6e3a1", "#f38ba8", "#fab387")
    for index, ((_, audio), ax) in enumerate(zip(items, axes)):
        data = _waveform(audio)
        time = np.arange(len(data)) / max(sample_rate, 1)
        ax.set_facecolor(theme["background"])
        ax.plot(time, data, color=colors[index % len(colors)], linewidth=0.8)
        ax.set_title(labels[index] if index < len(labels) else f"Waveform {index + 1}", color=theme["text"])
        ax.set_ylabel("Amplitude", color=theme["text"])
        ax.grid(True, color=theme["grid"], alpha=0.5, linestyle="--")
        ax.tick_params(colors=theme["text"])
    axes[-1].set_xlabel("Time (seconds)", color=theme["text"])
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_spectrograms(
    spectrograms: Mapping[str, AudioLike] | Sequence[AudioLike],
    sample_rate: int = 44100,
    hop_length: int = 256,
    titles: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dark_theme: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot one or more time-frequency representations and return ``(figure, axes)``."""
    items = list(spectrograms.items()) if isinstance(spectrograms, Mapping) else list(enumerate(spectrograms))
    if not items:
        raise ValueError("spectrograms must contain at least one spectrogram")
    labels = list(titles) if titles is not None else [str(label) for label, _ in items]
    theme = _theme(dark_theme)
    fig, axes = plt.subplots(len(items), 1, squeeze=False,
                             figsize=figsize or (12, max(3.0 * len(items), 4)))
    axes = axes[:, 0]
    fig.patch.set_facecolor(theme["figure"])
    for index, ((_, spec), ax) in enumerate(zip(items, axes)):
        array = spec.detach().cpu().numpy() if isinstance(spec, Tensor) else np.asarray(spec)
        while array.ndim > 2:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"Each spectrogram must be 2D, got {array.shape}")
        ax.imshow(array, origin="lower", aspect="auto", cmap="magma",
                  extent=(0, array.shape[1] * hop_length / max(sample_rate, 1), 0, array.shape[0]))
        ax.set_facecolor(theme["background"])
        ax.set_title(labels[index] if index < len(labels) else f"Spectrogram {index + 1}", color=theme["text"])
        ax.set_ylabel("Frequency bins", color=theme["text"])
        ax.tick_params(colors=theme["text"])
    axes[-1].set_xlabel("Time (seconds)", color=theme["text"])
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def display_audio_player(
    audio: AudioLike,
    sample_rate: int = 44100,
    title: str = "Audio",
    normalize: bool = True,
) -> Any:
    """Create and display an IPython audio player for a waveform."""
    try:
        from IPython.display import Audio, display
    except ImportError as error:
        raise RuntimeError("Audio players require IPython, which is normally available in notebooks.") from error
    data = _waveform(audio)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if normalize and peak > 1e-8:
        data = data * (0.9 / peak)
    player = Audio(data, rate=sample_rate)
    display(player)
    return player


def visualize_sample(
    sample: SampleLike,
    sample_idx: int = 0,
    sample_rate: Optional[int] = None,
    show_waveforms: bool = True,
    show_spectrograms: bool = True,
    show_player: bool = True,
    dark_theme: bool = True,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
) -> Dict[str, Any]:
    """Visualize a dataset sample in time, time-frequency, and audio domains.

    Missing residual audio or spectrograms are derived automatically. The
    returned dictionary contains NumPy audio arrays, spectrograms, figures, and
    an optional player, making it convenient for notebook inspection.
    """
    sr = sample_rate or int(_sample_value(_value(sample, "sample_rate", 44100), sample_idx) or 44100)
    mix = _waveform(_value(sample, "mix_audio"), sample_idx)
    target = _waveform(_value(sample, "target_audio"), sample_idx)
    residual_value = _value(sample, "residual_audio")
    residual = _waveform(residual_value, sample_idx) if residual_value is not None else mix - target
    audio = {"mix": mix, "target": target, "residual": residual}

    names = ("mix_spec", "target_spec", "residual_spec")
    specs: Dict[str, np.ndarray] = {}
    for name, key in zip(("mix", "target", "residual"), names):
        existing = _value(sample, key)
        if existing is not None:
            existing = _sample_value(existing, sample_idx)
            array = existing.detach().cpu().numpy() if isinstance(existing, Tensor) else np.asarray(existing)
            while array.ndim > 2:
                array = array[0]
            specs[name] = array
        else:
            specs[name] = _spectrogram(audio[name], sr, n_fft, hop_length, n_mels)

    track_id = _sample_value(_value(sample, "track_id", "sample"), sample_idx)
    stem_name = _sample_value(_value(sample, "stem_name", "target"), sample_idx)
    labels = [f"Mixture ({track_id})", f"Target ({stem_name})", "Residual"]
    result: Dict[str, Any] = {"audio": audio, "spectrograms": specs, "waveforms_fig": None, "spectrograms_fig": None, "player": None}
    if show_waveforms:
        result["waveforms_fig"], _ = plot_waveforms(audio, sr, labels, dark_theme=dark_theme)
    if show_spectrograms:
        result["spectrograms_fig"], _ = plot_spectrograms(specs, sr, hop_length, labels, dark_theme=dark_theme)
    if show_player:
        result["player"] = {key: display_audio_player(value, sr, title=label) for key, value, label in zip(audio, audio.values(), labels)}
    return result


def plot_sample(sample: SampleLike, **kwargs: Any) -> Dict[str, Any]:
    """Compatibility alias for :func:`visualize_sample`."""
    return visualize_sample(sample, **kwargs)


def display_sample_players(
    sample: SampleLike,
    sample_idx: int = 0,
    sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """Display mixture, target, and residual players for one dataset sample."""
    result = visualize_sample(
        sample,
        sample_idx=sample_idx,
        sample_rate=sample_rate,
        show_waveforms=False,
        show_spectrograms=False,
        show_player=True,
    )
    return result["player"]
