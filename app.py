"""ZeroGPU Gradio demo for Stable Audio 3 — Medium, Small Music, Small SFX.

All three models are preloaded at module level (per the ZeroGPU contract), and
a radio selector picks which one runs inside the ``@spaces.GPU`` infer call.
The visible UI mirrors the high-level ``stable_audio_3`` defaults (prompt +
duration); steps / CFG / sampler / seed live in an Advanced accordion.
"""

from __future__ import annotations

import spaces  # noqa: F401

import os
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass

def _ensure_stable_audio_tools() -> None:
    try:
        import stable_audio_tools  # noqa: F401
        return
    except ImportError:
        pass
    # stable-audio-tools 0.0.20 strict-pins torch==2.7.1 / torchaudio==2.7.1,
    # which lack sm_120 (Blackwell) kernels. Install with --no-deps; the
    # transitive deps are listed in requirements.txt and resolved against the
    # sm_120-capable torch at build time.
    print("[startup] installing stable-audio-tools (--no-deps) …", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
         "stable-audio-tools"],
    )
    import stable_audio_tools  # noqa: F401
    print("[startup] stable-audio-tools installed.", flush=True)

_ensure_stable_audio_tools()


import gradio as gr
import soundfile as sf
import torch
from einops import rearrange

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    key: str
    repo: str
    label: str
    default_duration: int
    placeholder: str


VARIANTS: list[Variant] = [
    Variant(
        key="medium",
        repo="stabilityai/stable-audio-3-medium",
        label="Medium — general audio (largest)",
        default_duration=60,
        placeholder="A dream-like Synthpop instrumental that would accompany a dream-sequence in a surrealist movie 120 BPM",
    ),
    Variant(
        key="small-music",
        repo="stabilityai/stable-audio-3-small-music",
        label="Small Music — 0.6B, music-focused",
        default_duration=60,
        placeholder="Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM",
    ),
    Variant(
        key="small-sfx",
        repo="stabilityai/stable-audio-3-small-sfx",
        label="Small SFX — 0.6B, sound effects",
        default_duration=7,
        placeholder="Chugging train coming into station with horn",
    ),
]


# ---------------------------------------------------------------------------
# Preload all variants at module level (ZeroGPU CUDA emulation accepts it)
# ---------------------------------------------------------------------------

@dataclass
class LoadedVariant:
    variant: Variant
    model: object
    sample_rate: int
    sample_size: int
    max_seconds: int


LOADED: dict[str, LoadedVariant] = {}
for v in VARIANTS:
    print(f"[startup] loading {v.repo} …", flush=True)
    t0 = time.time()
    model, config = get_pretrained_model(v.repo)
    sr = int(config["sample_rate"])
    ss = int(config["sample_size"])
    model = model.to("cuda").to(torch.float16)
    LOADED[v.key] = LoadedVariant(
        variant=v,
        model=model,
        sample_rate=sr,
        sample_size=ss,
        max_seconds=ss // sr,
    )
    print(
        f"[startup] {v.key} ready in {time.time() - t0:.1f}s · "
        f"sr={sr} · sample_size={ss} (~{ss // sr}s max)",
        flush=True,
    )

VARIANT_CHOICES = [(v.label, v.key) for v in VARIANTS]
SAMPLERS = ["pingpong", "k-dpmpp-2m", "k-heun", "dpmpp-2s-ancestral", "dpmpp-3m-sde"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@spaces.GPU(duration=180)
def infer(
    variant_key: str,
    prompt: str,
    duration: int = 60,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampler_type: str = "pingpong",
    seed: int = 0,
    progress: gr.Progress = gr.Progress(),
):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if variant_key not in LOADED:
        raise gr.Error(f"Unknown variant {variant_key!r}.")
    lv = LOADED[variant_key]

    duration = max(1, min(int(duration), lv.max_seconds))

    progress(0.1, desc=f"[{variant_key}] preparing conditioning")
    conditioning = [{"prompt": prompt, "seconds_total": int(duration)}]

    if seed and int(seed) > 0:
        torch.manual_seed(int(seed))
    else:
        torch.seed()

    progress(0.25, desc=f"[{variant_key}] sampling {steps} steps with {sampler_type}")
    t0 = time.time()
    output = generate_diffusion_cond_inpaint(
        lv.model,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        conditioning=conditioning,
        sample_size=lv.sample_size,
        sampler_type=sampler_type,
        device="cuda",
    )
    print(f"[infer/{variant_key}] sampling done in {time.time() - t0:.1f}s", flush=True)

    progress(0.92, desc="Normalising & saving")
    output = rearrange(output, "b d n -> d (b n)")
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)).clamp(min=1e-9))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()
    )
    output = output[:, : int(duration) * lv.sample_rate]

    out_path = os.path.join(tempfile.mkdtemp(), f"sa3_{variant_key}.wav")
    # soundfile expects (samples, channels); our tensor is (channels, samples).
    sf.write(out_path, output.numpy().T, lv.sample_rate, subtype="PCM_16")
    return out_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🎵 Stable Audio 3

Text-to-audio generation with [Stable Audio 3](https://huggingface.co/stabilityai). Pick a variant, write a prompt, hit Generate.
"""

EXAMPLES = [
    ["medium",      "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM", 60],
    ["small-music", "Cinematic neo-soul groove with electric piano, brushed drums, walking upright bass, smoky vibe 92 BPM", 45],
    ["small-music", "Driving techno track with rolling 16th-note hats, deep sub bass, acid arpeggios building tension 132 BPM", 60],
    ["small-sfx",   "Chugging train coming into station with horn", 7],
    ["small-sfx",   "Heavy rain on a tin roof with distant thunder rolls", 10],
    ["medium",      "Rainy night, lo-fi hip-hop beat with vinyl crackle, mellow piano chords, soft kick and snare 80 BPM", 30],
]


def _on_variant_change(variant_key: str):
    lv = LOADED[variant_key]
    return (
        gr.update(maximum=lv.max_seconds, value=min(lv.variant.default_duration, lv.max_seconds),
                  label=f"Duration (s) · model max {lv.max_seconds}s"),
        gr.update(placeholder=lv.variant.placeholder),
    )


with gr.Blocks(theme=gr.themes.Soft(), title="Stable Audio 3") as demo:
    gr.Markdown(DESCRIPTION)

    variant = gr.Radio(
        choices=VARIANT_CHOICES,
        value=VARIANTS[0].key,
        label="Model",
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder=VARIANTS[0].placeholder,
                lines=3,
            )
            duration = gr.Slider(
                1, LOADED[VARIANTS[0].key].max_seconds,
                value=VARIANTS[0].default_duration, step=1,
                label=f"Duration (s) · model max {LOADED[VARIANTS[0].key].max_seconds}s",
            )
            with gr.Accordion("Advanced settings", open=False):
                steps = gr.Slider(1, 50, value=8, step=1, label="Steps")
                cfg_scale = gr.Slider(0.5, 8.0, value=1.0, step=0.1, label="CFG scale")
                sampler_type = gr.Dropdown(SAMPLERS, value="pingpong", label="Sampler")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            run_btn = gr.Button("🎼 Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            audio_out = gr.Audio(label="Output", type="filepath", autoplay=True)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[variant, prompt, duration],
        outputs=[audio_out],
        fn=infer,
        cache_examples=True,
        cache_mode="lazy",
        label="Examples (lazy-cached on first click)",
    )

    variant.change(
        fn=_on_variant_change,
        inputs=[variant],
        outputs=[duration, prompt],
    )

    run_btn.click(
        fn=infer,
        inputs=[variant, prompt, duration, steps, cfg_scale, sampler_type, seed],
        outputs=[audio_out],
    )


if __name__ == "__main__":
    demo.launch()
