---
title: Stable Audio 3
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: true
license: other
short_description: Text-to-audio with SA3 Medium / Small Music / Small SFX.
suggested_hardware: zero-a10g
models:
- stabilityai/stable-audio-3-medium
- stabilityai/stable-audio-3-small-music
- stabilityai/stable-audio-3-small-sfx
---

# Stable Audio 3

ZeroGPU demo of the [Stable Audio 3](https://huggingface.co/stabilityai) family. Three variants preloaded at module load; switch between them with a radio button.

- [`stable-audio-3-medium`](https://huggingface.co/stabilityai/stable-audio-3-medium) — general audio (largest).
- [`stable-audio-3-small-music`](https://huggingface.co/stabilityai/stable-audio-3-small-music) — 0.6B, music-focused.
- [`stable-audio-3-small-sfx`](https://huggingface.co/stabilityai/stable-audio-3-small-sfx) — 0.6B, sound effects.
