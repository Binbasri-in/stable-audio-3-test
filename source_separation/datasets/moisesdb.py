"""MoisesDB source separation dataset."""

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

# 8.5 MoisesDB Multitrack Reader & Single-Source Dataset
# ---------------------------------------------------------------------------


class MoisesDBReader:
    """Cataloguer and reader for MoisesDB multitrack datasets.

    Supports the official MoisesDB hierarchical directory structure:
      data_path/
        <provider>/
          <track_id>/
            data.json
            <stem_name>/
              <source_id>.wav

    Computes composite mixtures dynamically by summing constituent stems.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None) -> None:
        if root_dir is None:
            default_paths = [
                DEFAULT_DATA_ROOT / "moisesdb" / "moisesdb_v0.1",
                DEFAULT_DATA_ROOT / "moisesdb",
            ]
            for p in default_paths:
                if p.exists():
                    root_dir = p
                    break
            if root_dir is None:
                root_dir = default_paths[0]
        self.root_dir = Path(root_dir).resolve()

    def scan_catalog(self, target_stem: Optional[str] = None) -> List[Dict[str, Any]]:
        """Scans catalog and returns metadata for tracks, optionally filtering by target stem."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"MoisesDB directory not found: {self.root_dir}")

        catalog = []
        for json_path in sorted(self.root_dir.glob("**/data.json")):
            track_dir = json_path.parent
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {json_path}: {e}")
                continue

            track_id = track_dir.name
            artist = meta.get("artist", "Unknown Artist")
            title = meta.get("song", meta.get("title", track_id))
            genre = meta.get("genre", "Unknown")

            stems_dict = {}
            for stem_entry in meta.get("stems", []):
                stem_name = stem_entry.get("stemName", "").lower()
                audio_paths = []
                for t in stem_entry.get("tracks", []):
                    ext = t.get("extension", "wav")
                    tid = t.get("id", stem_name)
                    audio_file = track_dir / stem_name / f"{tid}.{ext}"
                    if audio_file.exists():
                        audio_paths.append(audio_file)
                if audio_paths:
                    stems_dict[stem_name] = audio_paths

            if not stems_dict:
                continue

            if target_stem is not None:
                stem_clean = target_stem.strip().lower()
                if stem_clean not in stems_dict:
                    continue

            catalog.append({
                "track_id": track_id,
                "track_dir": track_dir,
                "artist": artist,
                "title": title,
                "genre": genre,
                "stems": stems_dict,
                "stem_names": list(stems_dict.keys()),
            })

        logger.info(f"MoisesDBReader scanned {len(catalog)} tracks from {self.root_dir}")
        return catalog


class MoisesDBSingleSourceDataset(Dataset):
    """Production PyTorch dataset for waveform Single-Source Separation using MoisesDB.

    MoisesDB provides 11 stem classes and detailed sub-instrument tracks.
    This dataset enables training waveform-domain single-source separation models (e.g. Demucs,
    Wave-U-Net, Conv-TasNet, or single-source generative priors) targeting any stem:
      - 'vocals', 'drums', 'bass', 'guitar', 'piano', 'other', etc.

    Preprocessing:
      - Direct physical pressure wave extraction [channels, samples]
      - Dynamic resampling to target_sr (e.g. 44.1 kHz, 24 kHz)
      - Channel formatting (mono or stereo)
      - Energy & Activity gating: rejects silent/inactive chunks using RMS energy threshold
      - Fixed-duration random or sequential chunking
      - Peak / RMS normalization
      - On-the-fly dynamic remixture augmentation
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        target_stem: Union[str, StemType] = "drums",
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
        self.reader = MoisesDBReader(root_dir=root_dir)
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

        # Scan catalog for tracks having target stem
        self.tracks = self.reader.scan_catalog(target_stem=self.target_stem_name)
        if not self.tracks:
            raise FileNotFoundError(
                f"No tracks found containing stem '{self.target_stem_name}' in {self.root_dir}"
            )
        logger.info(
            f"Initialized MoisesDBSingleSourceDataset: {len(self.tracks)} tracks containing "
            f"'{self.target_stem_name}'. Total virtual samples: {len(self)} (Duration: {self.segment_duration}s)."
        )

    def __len__(self) -> int:
        return len(self.tracks) * self.samples_per_track

    def _load_stem_audio(self, audio_paths: List[Path]) -> Tensor:
        """Loads and sums multiple sub-instrument audio files for a single stem."""
        loaded_audios = []
        for p in audio_paths:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)
            # data is [samples, channels] -> transpose to [channels, samples]
            t = torch.from_numpy(data.T)
            if sr != self.sample_rate:
                t = torchaudio.functional.resample(t, sr, self.sample_rate)
            if self.mono and t.shape[0] > 1:
                t = t.mean(dim=0, keepdim=True)
            elif not self.mono and t.shape[0] == 1:
                t = t.repeat(2, 1)
            loaded_audios.append(t)

        min_len = min(a.shape[-1] for a in loaded_audios)
        stem_audio = sum(a[..., :min_len] for a in loaded_audios)
        return stem_audio

    def __getitem__(self, index: int) -> SeparationPairSample:
        track_idx = (index // self.samples_per_track) % len(self.tracks)
        track_info = self.tracks[track_idx]

        # 1. Load target stem
        target_audio = self._load_stem_audio(track_info["stems"][self.target_stem_name])

        # 2. Load all other stems to construct composite mixture
        all_stems_audio = [target_audio]
        for stem_name, paths in track_info["stems"].items():
            if stem_name != self.target_stem_name:
                all_stems_audio.append(self._load_stem_audio(paths))

        min_len = min(a.shape[-1] for a in all_stems_audio)
        all_stems_aligned = [a[..., :min_len] for a in all_stems_audio]
        target_audio = all_stems_aligned[0]
        mix_audio = sum(all_stems_aligned)

        # 3. Dynamic Energy-gated slice
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

        # 4. Waveform Normalization
        mix_audio = self.preprocessor.normalize(mix_audio)
        target_audio = self.preprocessor.normalize(target_audio)
        residual_audio = mix_audio - target_audio

        # 5. Augmentation
        if self.augment and self.augmentor is not None:
            mix_audio, target_audio, residual_audio = self.augmentor(
                target_audio, residual_audio
            )

        # 6. Spectrograms & Priors (if requested)
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
