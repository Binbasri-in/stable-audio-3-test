from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 44100

def preprocess_audio_file(
    input_path: str | Path,
    output_path: str | Path,
    target_sr: int = SAMPLE_RATE,
    max_duration: float | None = None,
) -> tuple[int, torch.Tensor]:
    """Load WAV/MP3, standardize it, and save a normalized stereo WAV file."""
    input_path = str(input_path)
    output_path = Path(output_path)

    try:
        data, sr = sf.read(input_path, always_2d=False)
        data = np.asarray(data, dtype=np.float32)

        if data.ndim == 1:
            data = np.stack([data, data], axis=0)
        else:
            data = data[:, :2].T

        tensor = torch.from_numpy(data)

    except Exception:
        # Fallback for MP3 files when SoundFile lacks MP3 support.
        tensor, sr = torchaudio.load(input_path)
        tensor = tensor.float()

        if tensor.shape[0] == 1:
            tensor = tensor.repeat(2, 1)
        else:
            tensor = tensor[:2]

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
        )
        tensor = resampler(tensor)
        sr = target_sr

    if max_duration is not None:
        tensor = tensor[:, : int(max_duration * target_sr)]

    peak = tensor.abs().max()
    if peak > 0:
        tensor = tensor / peak * 0.90

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(output_path),
        tensor.detach().cpu().numpy().T,
        sr,
        subtype="PCM_16",
    )

    return sr, tensor

