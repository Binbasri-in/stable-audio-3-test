try:
    import spaces  # noqa: F401
except ImportError:
    class spaces:  # type: ignore
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None:
                return fn

            def decorator(f):
                return f

            return decorator

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from einops import rearrange
import gradio as gr
import soundfile as sf
import torch

from stable_audio import get_pretrained_model
from stable_audio.inference.generation import generate_diffusion_cond_inpaint


# ---- Load models -- #
SAMPLERS = ["pingpong", "euler", "rk4", "dpmpp"]
key="small-music"
repo="stabilityai/stable-audio-3-small-music"
label="Small Music — 0.6B, music-focused"
default_duration=60
placeholder="Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM"


print(f"[startup] loading {repo} …", flush=True)
t01 = time.time()
model, config = get_pretrained_model(repo)
sample_rate = int(config["sample_rate"])
sample_size = int(config["sample_size"])
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = model.to(device).to(dtype)
max_seconds = sample_size // sample_rate
print(
    f"[startup] {key} ready in {time.time() - t01:.1f}s · "
    f"sr={sample_rate} · sample_size={sample_size} (~{max_seconds}s max)",
    flush=True,
)

# ---- Inference -- #
def _tensor_to_wav(
    output: torch.Tensor,
    sample_rate: int,
    duration_seconds: Optional[int],
    out_dir: Optional[str] = None,
) -> Tuple[str, torch.Tensor]:
    """Pack a (B, C, N) generation tensor to int16, optionally cut to duration,
    write to disk, and return (path, int16-tensor)."""
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
    out_path = os.path.join(out_dir, "sa3.wav")
    sf.write(out_path, output.numpy().T, sample_rate, subtype="PCM_16")
    return out_path, output



def _run_inference(
    variant_key: str,
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    progress: gr.Progress = gr.Progress()
):
    prompt = (prompt or "").strip()

    duration = max(1, min(int(duration), max_seconds))

    progress(0.05, desc=f"[{variant_key}] preparing conditioning")
    conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]

    seed_val = int(seed) if seed and int(seed) > 0 else -1
    negative_conditioning = None
    sigma_max = 1.0
    apg_scale = 1.0
    duration_padding_sec = 6.0

    gen_kwargs: dict = dict(
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=sample_size,
        sampler_type=sampler_type,
        seed=seed_val,
        device=device,
        sigma_max=float(sigma_max),
        apg_scale=float(apg_scale),
        duration_padding_sec=float(duration_padding_sec),
    )

    progress(0.25, desc=f"[{variant_key}] sampling {steps} steps with {sampler_type}")
    t0 = time.time()
    output = generate_diffusion_cond_inpaint(model, **gen_kwargs)
    print(f"[infer/{variant_key}] sampling done in {time.time() - t0:.1f}s", flush=True)

    progress(0.92, desc="Normalising & saving")
    out_path, int16_audio = _tensor_to_wav(output, sample_rate, None)

    return out_path

@spaces.GPU
def infer(
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    progress: gr.Progress = gr.Progress(),
):
    """Slim handler used by the Simple tab and the Examples cache."""
    return _run_inference(
        variant_key="small-music",
        prompt=prompt,
        duration=duration,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler_type=sampler_type,
        seed=seed,
        progress=progress,
    )



# ---- UI -- #
DESCRIPTION = """
# 🎵 Stable Audio 3

Text-to-audio generation with <a href="https://huggingface.co/collections/stabilityai/stable-audio-3" target="_blank" rel="noopener noreferrer">Stable Audio 3</a>. Pick a variant, write a prompt, hit Generate. Switch to **Advanced** for the full sampler / init-audio / inpainting controls.
"""


with gr.Blocks(theme=gr.themes.Citrus(), title="Stable Audio 3") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder=placeholder,
                lines=3,
            )
            duration = gr.Slider(
                1, max_seconds,
                value=default_duration, step=1,
                label=f"Duration (s) · model max {max_seconds}s",
            )
            with gr.Accordion("Advanced settings", open=False):
                steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                cfg_scale = gr.Slider(0.5, 8.0, value=1.0, step=0.1, label="CFG scale")
                sampler_type = gr.Dropdown(SAMPLERS, value="pingpong", label="Sampler")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            run_btn = gr.Button("🎼 Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)

    # when clicking 
    run_btn.click(
        fn=infer,
        inputs=[prompt, duration, steps, cfg_scale, sampler_type, seed],
        outputs=[audio_out],
    )


if __name__ == "__main__":
    demo.launch()
