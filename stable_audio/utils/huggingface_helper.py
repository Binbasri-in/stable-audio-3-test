from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from .preprocess import preprocess_audio_file, SAMPLE_RATE


AUDIO_SAMPLES_DIR = Path(__file__).parent.parent / "audio_samples"
# ---------------------------------------------------------------------------
# Curated Catalog of Real Musical Recordings Available Online
# ---------------------------------------------------------------------------
REAL_MUSIC_CATALOG = {
    "acoustic_guitar": {
        "title": "Acoustic Guitar Fingerpicking & Chords",
        "repo_id": "cchaun/musical_instrument_identification",
        "filename": "samples/guitar_acoustic.wav",
        "repo_type": "space",
        "description": "Authentic acoustic guitar chord progression, 16s stereo 44.1kHz",
        "recommended_inpaint_prompt": "Soulful blues electric lead guitar solo over acoustic rhythm",
    },
    "piano": {
        "title": "Grand Piano Classical Progression",
        "repo_id": "cchaun/musical_instrument_identification",
        "filename": "samples/piano.wav",
        "repo_type": "space",
        "description": "Rich polyphonic acoustic grand piano, 16s stereo 44.1kHz",
        "recommended_inpaint_prompt": "Warm cello melody harmonizing with acoustic piano chords",
    },
    "electric_guitar": {
        "title": "Electric Guitar Rock Riff",
        "repo_id": "cchaun/musical_instrument_identification",
        "filename": "samples/guitar_electric.wav",
        "repo_type": "space",
        "description": "Overdriven electric guitar rock riff, ~20s stereo 44.1kHz",
        "recommended_inpaint_prompt": "Heavy rock drum fill with double-kick and crash cymbals",
    },
    "electronic_beat": {
        "title": "Electronic Synth & Bass Groove",
        "repo_id": "facebook/MusicGen",
        "filename": "dataset/example/electro_1.mp3",
        "repo_type": "space",
        "description": "Punchy analog synth arpeggio and electronic sub-bass, 15s stereo",
        "recommended_inpaint_prompt": "Funky vocoder vocal chop with bright disco synth stab",
    },
    "full_band_rock": {
        "title": "Full Band Indie Rock & Vocals",
        "repo_id": "abidlabs/music-separation",
        "filename": "test.mp3",
        "repo_type": "space",
        "description": "Full band instrumentation (drums, bass, guitar, vocals), 10.9s stereo",
        "recommended_inpaint_prompt": "Ripping fuzz guitar solo leading into an energetic drum crash",
    },
}


def load_or_download_real_music(
    track_key: str = "acoustic_guitar",
    target_sr: int = SAMPLE_RATE,
    max_duration: float | None = None,
    out_dir: Path = AUDIO_SAMPLES_DIR,
) -> tuple[str, tuple[int, torch.Tensor]]:
    """Download a catalog track if needed, preprocess it, and return its audio."""
    if track_key not in REAL_MUSIC_CATALOG:
        raise ValueError(
            f"Unknown track '{track_key}'. "
            f"Available: {list(REAL_MUSIC_CATALOG.keys())}"
        )

    meta = REAL_MUSIC_CATALOG[track_key]
    output_path = out_dir / f"{track_key}.wav"

    if output_path.exists():
        input_path = output_path
    else:
        print(f"Downloading real music track '{meta['title']}'...")
        input_path = hf_hub_download(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            repo_type=meta["repo_type"],
        )

    sr, tensor = preprocess_audio_file(
        input_path=input_path,
        output_path=output_path,
        target_sr=target_sr,
        max_duration=max_duration,
    )

    print(f"✓ Loaded real music: '{meta['title']}'")
    print(f"   • Path: {output_path}")
    print(
        f"   • Duration: {tensor.shape[1] / sr:.2f}s"
        f" | Sample Rate: {sr} Hz"
        f" | Channels: {tensor.shape[0]}"
    )

    return str(output_path), (sr, tensor)