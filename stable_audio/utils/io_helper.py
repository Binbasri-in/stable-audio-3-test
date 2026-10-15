import os
import tempfile
import torch
import soundfile as sf
from einops import rearrange

def tensor_to_wav(
    output: torch.Tensor,
    sample_rate: int,
    duration_seconds: int = None,
    out_dir: str = None,
    filename: str = "sa3.wav",
) -> tuple[str, torch.Tensor]:
    """Pack a (B, C, N) generation tensor to int16, optionally cut to duration,
    write to disk, and return (path, int16-tensor)."""
    if output.ndim == 1:
        output = output.unsqueeze(0).unsqueeze(0)
    elif output.ndim == 2:
        output = output.unsqueeze(0)
    output = rearrange(output, "b d n -> d (b n)")
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)).clamp(min=1e-9))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    if duration_seconds is not None:
        output = output[:, : int(duration_seconds) * sample_rate]
    out_dir = out_dir or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    sf.write(out_path, output.numpy().T, sample_rate, subtype="PCM_16")
    return out_path, output
