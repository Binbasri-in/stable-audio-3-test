"""Slakh2100 source separation dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from .common import (
    AcousticPriorExtractor,
    AudioPreprocessor,
    DEFAULT_DATA_ROOT,
    SeparationAugmentor,
    SeparationPairSample,
    StemType,
    find_active_time_offsets,
    logger,
)

# 8.6 Slakh2100 Reader & Single-Source Dataset (Guitar, Piano, etc.)
# ---------------------------------------------------------------------------


class SlakhReader:
    """Cataloguer and reader for the Synthesized Lakh (Slakh2100) dataset.

    Slakh2100 provides 2,100 studio-grade synthesized multitracks with zero
    acoustic bleed, making it ideal for learning exact instrument generative
    priors (e.g. Normalizing Flows like Glow) for pure instruments like GUITAR and PIANO.

    Directory layout:
      root_dir/
        Track00001/
          mix.wav (or mix.flac)
          metadata.json / metadata.yaml
          stems/
            S00.wav (e.g. Piano)
            S01.wav (e.g. Guitar)
            S02.wav (e.g. Bass)
            S03.wav (e.g. Drums)
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        if root_dir is None:
            default_paths = [
                DEFAULT_DATA_ROOT / "slakh2100",
                DEFAULT_DATA_ROOT / "slakh",
            ]
            for p in default_paths:
                if p.exists():
                    root_dir = p
                    break
            if root_dir is None:
                root_dir = default_paths[0]
        self.root_dir = Path(root_dir).resolve()

    def scan_catalog(
        self, target_instrument: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Scans catalog, filtering by instrument class (e.g., 'guitar', 'piano')."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Slakh2100 directory not found: {self.root_dir}")

        catalog = []
        track_dirs = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith("Track")]
        )
        target_clean = target_instrument.strip().lower() if target_instrument else None

        for t_dir in track_dirs:
            meta_json = t_dir / "metadata.json"
            meta_yaml = t_dir / "metadata.yaml"
            stems_dir = t_dir / "stems"
            mix_path = t_dir / "mix.wav"
            if not mix_path.exists():
                mix_path = t_dir / "mix.flac"

            stems_meta = {}
            if meta_json.exists():
                try:
                    with open(meta_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    stems_meta = data.get("stems", {})
                except Exception as e:
                    logger.warning(f"Failed to read {meta_json}: {e}")
            elif meta_yaml.exists():
                try:
                    with open(meta_yaml, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    current_stem = None
                    for line in lines:
                        line_s = line.strip()
                        if line_s.startswith("S") and line_s.endswith(":"):
                            current_stem = line_s[:-1]
                            stems_meta[current_stem] = {}
                        elif current_stem and ":" in line_s:
                            k, v = line_s.split(":", 1)
                            stems_meta[current_stem][k.strip()] = v.strip().strip("'\"")
                except Exception as e:
                    logger.warning(f"Failed to parse {meta_yaml}: {e}")

            # Map instrument classes to audio paths
            inst_paths: Dict[str, List[Path]] = {}
            for stem_id, s_info in stems_meta.items():
                inst_class = str(s_info.get("inst_class", "")).lower()
                fname = s_info.get("file_name", f"{stem_id}.wav")
                fpath = stems_dir / fname
                if not fpath.exists():
                    fpath = stems_dir / f"{stem_id}.flac"
                if fpath.exists():
                    inst_paths.setdefault(inst_class, []).append(fpath)

            if not inst_paths:
                continue

            if target_clean is not None:
                matching_keys = [k for k in inst_paths.keys() if target_clean in k]
                if not matching_keys:
                    continue

            catalog.append({
                "track_id": t_dir.name,
                "track_dir": t_dir,
                "mix_path": mix_path if mix_path.exists() else None,
                "stems": inst_paths,
                "instruments": list(inst_paths.keys()),
            })

        logger.info(f"SlakhReader scanned {len(catalog)} tracks from {self.root_dir}")
        return catalog


class SlakhSingleSourceDataset(Dataset):
    """Production PyTorch dataset for waveform Single-Source Separation using Slakh2100.

    Slakh2100 provides studio-grade synthesized multitracks with zero acoustic bleed,
    enabling exact generative prior learning (e.g. Normalizing Flows like Glow or
    WaveGlow) for pure acoustic instruments like GUITAR and PIANO.

    Features:
      - Clean single-source waveform extraction [channels, samples]
      - Anti-aliased resampling to target sample_rate (e.g. 44.1 kHz, 16 kHz)
      - Energy & Activity gating: rejects silent/unplayed chunks
      - Exact physical residual accompaniment: residual = mix - target
      - Normalization: peak or RMS scaling to [-1.0, 1.0]
      - Optional dynamic remixture augmentation
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        target_stem: Union[str, StemType] = "guitar",
        segment_duration: float = 10.0,
        sample_rate: int = 44100,
        mono: bool = True,
        return_spectrogram: bool = False,
        return_residual: bool = True,
        compute_priors: bool = False,
        augment: bool = False,
        samples_per_track: int = 5,
        energy_threshold: float = 0.005,
        min_active_ratio: float = 0.5,
        random_crop: bool = True,
        n_mels: int = 64,
        hop_length: int = 256,
        n_fft: int = 1024,
    ) -> None:
        super().__init__()
        self.reader = SlakhReader(root_dir=root_dir)
        self.root_dir = self.reader.root_dir

        if isinstance(target_stem, StemType):
            self.target_stem_name = target_stem.display_name
            self.target_stem_type = target_stem
            self.target_stem_id = int(target_stem)
        else:
            self.target_stem_name = str(target_stem).strip().lower()
            try:
                self.target_stem_type = StemType.from_str(self.target_stem_name)
                self.target_stem_id = int(self.target_stem_type)
            except Exception:
                self.target_stem_type = None
                self.target_stem_id = 99

        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.mono = mono
        self.return_spectrogram = return_spectrogram
        self.return_residual = return_residual
        self.compute_priors = compute_priors
        self.augment = augment
        self.samples_per_track = max(1, samples_per_track)
        self.energy_threshold = energy_threshold
        self.min_active_ratio = min_active_ratio
        self.random_crop = random_crop

        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            mono=mono,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.augmentor = SeparationAugmentor() if augment else None
        self.target_samples = int(self.segment_duration * self.sample_rate)

        # Scan catalog for tracks with matching instrument
        self.tracks = self.reader.scan_catalog(target_instrument=self.target_stem_name)
        if not self.tracks:
            raise FileNotFoundError(
                f"No tracks found containing instrument '{self.target_stem_name}' in {self.root_dir}"
            )
        logger.info(
            f"Initialized SlakhSingleSourceDataset: {len(self.tracks)} tracks containing "
            f"'{self.target_stem_name}'. Total virtual samples: {len(self)} (Duration: {self.segment_duration}s)."
        )

    def __len__(self) -> int:
        return len(self.tracks) * self.samples_per_track

    def _load_audio_file(self, path: Path) -> Tensor:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        t = torch.from_numpy(data.T)
        if sr != self.sample_rate:
            t = torchaudio.functional.resample(t, sr, self.sample_rate)
        if self.mono and t.shape[0] > 1:
            t = t.mean(dim=0, keepdim=True)
        elif not self.mono and t.shape[0] == 1:
            t = t.repeat(2, 1)
        return t

    def __getitem__(self, index: int) -> SeparationPairSample:
        track_idx = (index // self.samples_per_track) % len(self.tracks)
        track_info = self.tracks[track_idx]

        # Find matching stem keys for target_instrument
        target_keys = [k for k in track_info["stems"].keys() if self.target_stem_name in k]
        target_paths = track_info["stems"][target_keys[0]]
        target_audio = sum(self._load_audio_file(p) for p in target_paths)

        # If mix.wav is available, load it; otherwise sum all stems
        if track_info["mix_path"] and track_info["mix_path"].exists():
            mix_audio = self._load_audio_file(track_info["mix_path"])
        else:
            all_stems = [target_audio]
            for inst, paths in track_info["stems"].items():
                if inst not in target_keys:
                    all_stems.append(sum(self._load_audio_file(p) for p in paths))
            min_len = min(a.shape[-1] for a in all_stems)
            mix_audio = sum(a[..., :min_len] for a in all_stems)

        # Align length
        min_len = min(target_audio.shape[-1], mix_audio.shape[-1])
        target_audio = target_audio[..., :min_len]
        mix_audio = mix_audio[..., :min_len]

        # Dynamic Energy-gated slice
        total_samples = target_audio.shape[-1]
        if total_samples <= self.target_samples:
            pad_len = self.target_samples - total_samples
            mix_audio = torch.nn.functional.pad(mix_audio, (0, pad_len))
            target_audio = torch.nn.functional.pad(target_audio, (0, pad_len))
            start_sec = 0.0
        else:
            cand_offsets = find_active_time_offsets(
                target_audio=target_audio,
                needed_samples=self.target_samples,
                sample_rate=self.sample_rate,
                energy_threshold=self.energy_threshold,
                min_active_ratio=self.min_active_ratio,
            )
            if self.random_crop:
                start_idx = random.choice(cand_offsets)
            else:
                sub_idx = index % self.samples_per_track
                step_idx = min(
                    int(sub_idx * len(cand_offsets) / self.samples_per_track),
                    len(cand_offsets) - 1,
                )
                start_idx = cand_offsets[step_idx]

            mix_audio = mix_audio[:, start_idx : start_idx + self.target_samples]
            target_audio = target_audio[:, start_idx : start_idx + self.target_samples]
            start_sec = start_idx / self.sample_rate

        # Normalization
        mix_audio = self.preprocessor.normalize(mix_audio)
        target_audio = self.preprocessor.normalize(target_audio)
        residual_audio = mix_audio - target_audio

        # Augmentation
        if self.augment and self.augmentor is not None:
            mix_audio, target_audio, residual_audio = self.augmentor(
                target_audio, residual_audio
            )

        # Spectrograms & Priors (if requested)
        mix_spec, target_spec, res_spec = None, None, None
        if self.return_spectrogram:
            mix_spec = self.preprocessor.compute_spectrogram(mix_audio)
            target_spec = self.preprocessor.compute_spectrogram(target_audio)
            if self.return_residual:
                res_spec = self.preprocessor.compute_spectrogram(residual_audio)

        priors = {}
        if self.compute_priors and target_spec is not None:
            priors = AcousticPriorExtractor.extract_all(
                target_spec=target_spec,
                stem_type=self.target_stem_type or StemType.OTHER,
            )

        return SeparationPairSample(
            mix_audio=mix_audio,
            target_audio=target_audio,
            residual_audio=residual_audio if self.return_residual else None,
            mix_spec=mix_spec,
            target_spec=target_spec,
            residual_spec=res_spec,
            stem_type=self.target_stem_type or StemType.OTHER,
            stem_name=self.target_stem_name,
            stem_id=self.target_stem_id,
            sample_rate=self.sample_rate,
            track_id=track_info["track_id"],
            start_time=start_sec,
            duration=self.segment_duration,
            priors=priors,
        )


# ---------------------------------------------------------------------------
