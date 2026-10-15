import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display
import torch
import torchaudio.transforms as T

def compute_mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_mels: int = 128) -> np.ndarray:
    """Compute log Mel spectrogram for the left/mono channel."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    n_fft = 1024
    hop_length = n_fft // 2
    mel_op = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=None,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )
    melspec = mel_op(waveform.float())[0]  # left channel
    amin = 1e-10
    mel_db = 10.0 * np.log10(np.maximum(amin, melspec.numpy()))
    return mel_db


def play_and_display(wav_path, *args, title: str = "Generated Audio", **kwargs):
    """Render audio player, waveform, and spectrogram in the notebook."""
    if isinstance(wav_path, (tuple, list)):
        wav_path = wav_path[0]
    if args and isinstance(args[0], str):
        title = args[0]
    info = sf.info(wav_path)
    data, sr = sf.read(wav_path)
    
    print(f"\n🔊 [{title}]")
    print(f"   • Duration: {info.duration:.2f}s | Channels: {info.channels} | Sample Rate: {info.samplerate} Hz")
    
    # In-notebook audio playback
    display(Audio(wav_path))
    
    # Plot Waveform & Mel Spectrogram side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.5), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # Waveform plot
    time_axis = np.linspace(0, info.duration, len(data))
    if data.ndim > 1:
        ax1.plot(time_axis, data[:, 0], color="#2b5c8f", alpha=0.8, label="Left")
        ax1.plot(time_axis, data[:, 1], color="#e07a5f", alpha=0.6, label="Right")
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.plot(time_axis, data, color="#2b5c8f", alpha=0.8)
    ax1.set_title(f"{title} — Waveform", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # Spectrogram plot
    waveform_tensor = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
    mel_db = compute_mel_spectrogram(waveform_tensor, sr)
    im = ax2.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=np.percentile(mel_db, 5),   # Ignore extreme floor noise
            vmax=np.max(mel_db)
        )
    ax2.set_title(f"{title} — Mel Spectrogram", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mel Frequency Bins")
    ax2.set_xlabel("Time Frames")
    plt.colorbar(im, ax=ax2, label="dB")
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)