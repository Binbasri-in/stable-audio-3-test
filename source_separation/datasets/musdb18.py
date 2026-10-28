"""MUSDB18 source separation dataset."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .common import (
    AcousticPriorExtractor,
    AudioPreprocessor,
    SeparationAugmentor,
    SeparationPairSample,
    StemType,
    TrackMetadata,
    logger,
)

# 6. MUSDB18 Fast Stem Reader
# ---------------------------------------------------------------------------


def _resolve_musdb_dir(root_dir: str | Path | None) -> Path:
    """Robustly locates the musdb18 root directory across various execution working directories."""
    if root_dir is not None:
        p = Path(root_dir)
        if p.exists() and (p / "train").exists():
            return p.resolve()
        for cand in [
            Path.cwd() / p,
            Path(__file__).resolve().parents[2] / p,
        ]:
            if cand.exists() and (cand / "train").exists():
                return cand.resolve()

    for default_cand in [
        Path(__file__).resolve().parents[2] / "data" / "musdb18",
        Path.cwd() / "data" / "musdb18",
    ]:
        if default_cand.exists() and (default_cand / "train").exists():
            return default_cand.resolve()

    return (Path(__file__).resolve().parents[2] / "data" / "musdb18").resolve()


class MUSDB18Reader:
    """High-performance cataloguer and reader for MUSDB18 .stem.mp4 files.

    Features:
      - Uses `stempeg` for fast random time-interval decoding (`start`, `duration`).
      - Only decodes requested streams (e.g. mixture + target stem) to minimize I/O overhead.
      - Disk caching for fast track index initialization.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = _resolve_musdb_dir(root_dir)
        self.cache_path = self.root_dir / ".musdb18_catalog_cache.json"
        self._check_dependencies()

    @staticmethod
    def _check_dependencies() -> None:
        try:
            import stempeg  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "stempeg is required to read MUSDB18 .stem.mp4 files. "
                "Install it using: uv pip install stempeg"
            ) from e

    def scan_split(self, split: str = "train") -> list[TrackMetadata]:
        """Scans specified split directory ('train' or 'test') and catalogs tracks."""
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}. "
                f"Expected musdb18/{split} with .stem.mp4 files."
            )

        # Check existing cache
        cache = self._load_cache()
        if split in cache and len(cache[split]) > 0:
            tracks = []
            for item in cache[split]:
                f_path = Path(item["file_path"])
                if not f_path.exists():
                    f_path = (split_dir / f_path.name).resolve()
                else:
                    f_path = f_path.resolve()
                tracks.append(
                    TrackMetadata(
                        track_id=item["track_id"],
                        file_path=f_path,
                        artist=item["artist"],
                        title=item["title"],
                        duration=item["duration"],
                        sample_rate=item["sample_rate"],
                        channels=item["channels"],
                        stem_activity=item.get("stem_activity", {}),
                    )
                )
            logger.info(
                f"Loaded {len(tracks)} tracks from cache for '{split}' split."
            )
            return tracks

        # Scan files and parse metadata
        stem_files = sorted(split_dir.glob("*.stem.mp4"))
        if not stem_files:
            raise FileNotFoundError(
                f"No .stem.mp4 files found in {split_dir}."
            )

        import stempeg

        logger.info(
            f"Cataloging {len(stem_files)} tracks in {split_dir}... (one-time indexing)"
        )
        tracks = []
        cache_items = []

        for fpath in stem_files:
            track_name = fpath.stem.replace(".stem", "")
            if " - " in track_name:
                artist, title = track_name.split(" - ", 1)
            else:
                artist, title = "", track_name

            try:
                info = stempeg.Info(str(fpath))
                dur = float(info.duration(0))
                sr = int(info.rate(0))
                ch = int(info.channels(0))
            except Exception as ex:
                logger.warning(
                    f"Failed to read info for {fpath.name}: {ex}. Using defaults."
                )
                dur = 180.0
                sr = 44100
                ch = 2

            meta = TrackMetadata(
                track_id=track_name,
                file_path=fpath,
                artist=artist.strip(),
                title=title.strip(),
                duration=dur,
                sample_rate=sr,
                channels=ch,
            )
            tracks.append(meta)
            cache_items.append(
                {
                    "track_id": meta.track_id,
                    "file_path": str(meta.file_path),
                    "artist": meta.artist,
                    "title": meta.title,
                    "duration": meta.duration,
                    "sample_rate": meta.sample_rate,
                    "channels": meta.channels,
                    "stem_activity": meta.stem_activity,
                }
            )

        cache[split] = cache_items
        self._save_cache(cache)
        logger.info(
            f"Cataloged and cached {len(tracks)} tracks for '{split}' split."
        )
        return tracks

    def get_track_stem_activity(
        self, track: TrackMetadata, stem_name: str
    ) -> list[float]:
        """Returns 1-second RMS activity profile for the given stem, computing and caching if needed."""
        if stem_name in track.stem_activity and len(track.stem_activity[stem_name]) > 0:
            return track.stem_activity[stem_name]

        stem_type = StemType.from_str(stem_name)
        stem_id = int(stem_type)
        try:
            import stempeg
            f_p = track.file_path
            if not f_p.exists():
                for cand in [
                    self.root_dir / "train" / f_p.name,
                    self.root_dir / "test" / f_p.name,
                    self.root_dir / f_p.name,
                ]:
                    if cand.exists():
                        f_p = cand.resolve()
                        break
            stems, sr = stempeg.read_stems(
                str(f_p.resolve()),
                stem_id=[stem_id],
                dtype=np.float32,
            )
            audio = stems[0] if stems.ndim == 3 else stems
            num_blocks = audio.shape[0] // sr
            if num_blocks > 0:
                blocks = audio[: num_blocks * sr].reshape(num_blocks, -1)
                rms_vals = np.sqrt(np.mean(blocks ** 2, axis=1))
                profile = [round(float(v), 5) for v in rms_vals]
            else:
                profile = [round(float(np.sqrt(np.mean(audio ** 2))), 5)]
        except Exception as e:
            logger.warning(
                f"Could not compute stem activity for {track.track_id} ({stem_name}): {e}"
            )
            profile = []

        track.stem_activity[stem_name] = profile
        return profile

    def read_stem_slice(
        self,
        file_path: str | Path,
        start_sec: float,
        duration_sec: float,
        target_stem: StemType,
    ) -> tuple[Tensor, Tensor]:
        """Reads a synchronized time segment for mixture and the single target stem.

        Args:
            file_path: Path to .stem.mp4 file.
            start_sec: Offset in seconds.
            duration_sec: Length of segment in seconds.
            target_stem: Desired stem (DRUMS=1, BASS=2, OTHER=3, VOCALS=4).

        Returns:
            Tuple[Tensor, Tensor]:
                mix_audio: Shape [channels, samples]
                target_audio: Shape [channels, samples]
        """
        import stempeg

        stem_id = int(target_stem)
        f_p = Path(file_path)
        if not f_p.exists():
            for cand in [
                self.root_dir / "train" / f_p.name,
                self.root_dir / "test" / f_p.name,
                self.root_dir / f_p.name,
            ]:
                if cand.exists():
                    f_p = cand.resolve()
                    break

        # Read only mixture (stream 0) and target stem (stream target_stem)
        stems, sr = stempeg.read_stems(
            str(f_p.resolve()),
            start=max(0.0, start_sec),
            duration=duration_sec,
            stem_id=[0, stem_id],
            dtype=np.float32,
        )

        # stems shape is (2, samples, channels)
        # stems[0] -> mixture, stems[1] -> target stem
        mix = torch.from_numpy(stems[0]).t()  # [channels, samples]
        target = torch.from_numpy(stems[1]).t()  # [channels, samples]
        return mix, target

    def _load_cache(self) -> dict[str, Any]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict[str, Any]) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache to {self.cache_path}: {e}")


# ---------------------------------------------------------------------------
# 7. MUSDB18 Single Source Separation Dataset Class
# ---------------------------------------------------------------------------


class MUSDB18SingleSourceDataset(Dataset):
    """PyTorch Dataset for Single Source Separation and Style Prior Network Training.

    Extracts synchronized pairs of:
      1) Composite musical mixture (x)
      2) Isolated single instrument target (s_k) [Drums, Bass, Vocals, or Other]
      3) Accompaniment residual (r = x - s_k)
      4) 2D Log-Mel spectrograms matching AudioSPN architecture
      5) Structured musical priors (Rhythm Flux, Harmonic Chroma, Melodic Centroid)
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        split: str = "train",
        target_stem: str | StemType = "drums",
        segment_duration: float = 10.0,
        sample_rate: int = 44100,
        mono: bool = True,
        return_spectrogram: bool = True,
        return_residual: bool = True,
        compute_priors: bool = True,
        augment: bool = False,
        samples_per_track: int = 10,
        energy_threshold: float = 0.005,
        min_active_ratio: float = 0.5,
        random_crop: bool = True,
        n_mels: int = 64,
        hop_length: int = 256,
        n_fft: int = 1024,
        patch_frames: int | None = None,
        max_tracks: int | None = None,
        cache_in_memory: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = _resolve_musdb_dir(root_dir)
        self.split = split.lower()
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
        self.cache_in_memory = cache_in_memory
        self._memory_cache: dict[int, SeparationPairSample] = {}

        # Target stem configuration
        self.target_stem_str = str(target_stem).lower()
        if self.target_stem_str == "random":
            self.fixed_stem = None
        else:
            self.fixed_stem = StemType.from_str(target_stem)

        # Initialize helper modules
        self.reader = MUSDB18Reader(self.root_dir)
        self.tracks = self.reader.scan_split(self.split)
        if not self.tracks:
            raise RuntimeError(
                f"No tracks found in {self.root_dir} for split '{self.split}'."
            )

        # Exclude tracks where the target stem has 0 active signal throughout the entire song
        if self.fixed_stem is not None:
            filtered_tracks = []
            stem_key = self.fixed_stem.display_name
            for t in self.tracks:
                if stem_key in t.stem_activity and len(t.stem_activity[stem_key]) > 0:
                    max_stem_rms = max(t.stem_activity[stem_key])
                    if max_stem_rms >= self.energy_threshold:
                        filtered_tracks.append(t)
                else:
                    filtered_tracks.append(t)
            if filtered_tracks:
                if len(filtered_tracks) < len(self.tracks):
                    logger.info(
                        f"Filtered {len(self.tracks) - len(filtered_tracks)} tracks lacking active '{stem_key}'. "
                        f"{len(filtered_tracks)} tracks remaining."
                    )
                self.tracks = filtered_tracks

        if max_tracks is not None:
            self.tracks = self.tracks[:max(1, max_tracks)]

        self.preprocessor = AudioPreprocessor(
            sample_rate=sample_rate,
            mono=mono,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            patch_frames=patch_frames,
        )

        self.augmentor = SeparationAugmentor() if augment else None
        self.target_samples = int(self.segment_duration * self.sample_rate)

        logger.info(
            f"Initialized MUSDB18SingleSourceDataset ({self.split}): "
            f"{len(self.tracks)} tracks x {self.samples_per_track} samples = {len(self)} total items. "
            f"Target stem: {self.target_stem_str.upper()}, Duration: {self.segment_duration}s @ {self.sample_rate}Hz "
            f"(Silence excluded, energy_threshold: {self.energy_threshold})."
        )

    def __len__(self) -> int:
        return len(self.tracks) * self.samples_per_track

    def _select_stem(self) -> StemType:
        """Determines which instrument to isolate for the current sample."""
        if self.fixed_stem is not None:
            return self.fixed_stem
        # Random instrument stem from the 4 primary types
        return random.choice(
            [StemType.DRUMS, StemType.BASS, StemType.OTHER, StemType.VOCALS]
        )

    def _determine_start_time(
        self, track: TrackMetadata, stem_type: StemType, idx: int
    ) -> float:
        """Calculates start offset ensuring target stem is active (non-silent)."""
        max_start = max(0.0, track.duration - self.segment_duration)
        if max_start <= 0.0:
            return 0.0

        stem_name = stem_type.display_name
        profile = self.reader.get_track_stem_activity(track, stem_name)

        win_sec = max(1, int(self.segment_duration))
        valid_starts = []
        if profile and len(profile) >= win_sec:
            max_p_start = min(len(profile) - win_sec, int(max_start))
            for t in range(max_p_start + 1):
                win_slice = profile[t : t + win_sec]
                win_rms = math.sqrt(sum(v**2 for v in win_slice) / len(win_slice))
                act_ratio = sum(1 for v in win_slice if v >= self.energy_threshold) / len(win_slice)
                if win_rms >= self.energy_threshold and act_ratio >= self.min_active_ratio:
                    valid_starts.append(t)

        if valid_starts:
            if self.random_crop:
                base_t = float(random.choice(valid_starts))
                jitter = random.uniform(0.0, min(1.0, max(0.0, max_start - base_t)))
                return min(base_t + jitter, max_start)
            else:
                sub_idx = idx % self.samples_per_track
                step_idx = min(int(sub_idx * len(valid_starts) / self.samples_per_track), len(valid_starts) - 1)
                return float(valid_starts[step_idx])

        # If no window met full criteria, pick window with highest RMS
        if profile and len(profile) >= win_sec:
            max_p_start = min(len(profile) - win_sec, int(max_start))
            best_t = 0
            best_rms = -1.0
            for t in range(max_p_start + 1):
                win_slice = profile[t : t + win_sec]
                w_rms = sum(v**2 for v in win_slice)
                if w_rms > best_rms:
                    best_rms = w_rms
                    best_t = t
            return float(best_t)

        # Fallback if profile unavailable
        if not self.random_crop:
            sub_idx = idx % self.samples_per_track
            step = max_start / max(1, self.samples_per_track)
            return min(sub_idx * step, max_start)
        return random.uniform(0.0, max_start)

    def __getitem__(self, idx: int) -> SeparationPairSample:
        if self.cache_in_memory and idx in self._memory_cache:
            return self._memory_cache[idx]

        track_idx = idx // self.samples_per_track
        track = self.tracks[track_idx]
        stem_type = self._select_stem()
        start_time = self._determine_start_time(track, stem_type, idx)

        # 1. Read slice from .stem.mp4
        try:
            mix_raw, target_raw = self.reader.read_stem_slice(
                file_path=track.file_path,
                start_sec=start_time,
                duration_sec=self.segment_duration,
                target_stem=stem_type,
            )
        except Exception as e:
            logger.warning(
                f"Error loading {track.track_id} at {start_time:.1f}s: {e}. Falling back to 0.0s."
            )
            mix_raw, target_raw = self.reader.read_stem_slice(
                file_path=track.file_path,
                start_sec=0.0,
                duration_sec=self.segment_duration,
                target_stem=stem_type,
            )
            start_time = 0.0

        # Safety gate: verify sliced target stem has active energy; probe if silent
        target_rms = float(torch.sqrt(torch.mean(target_raw ** 2)))
        max_start = max(0.0, track.duration - self.segment_duration)
        if target_rms < self.energy_threshold and max_start > 0.0:
            cand_offsets = [0.25 * max_start, 0.5 * max_start, 0.75 * max_start, 0.33 * max_start, 0.66 * max_start]
            best_t = start_time
            best_rms = target_rms
            best_mix = mix_raw
            best_target = target_raw
            for cand in cand_offsets:
                if abs(cand - start_time) > 1.0:
                    try:
                        m_c, t_c = self.reader.read_stem_slice(
                            file_path=track.file_path,
                            start_sec=cand,
                            duration_sec=self.segment_duration,
                            target_stem=stem_type,
                        )
                        c_rms = float(torch.sqrt(torch.mean(t_c ** 2)))
                        if c_rms > best_rms:
                            best_rms = c_rms
                            best_mix = m_c
                            best_target = t_c
                            best_t = cand
                        if c_rms >= self.energy_threshold:
                            break
                    except Exception:
                        pass
            mix_raw, target_raw = best_mix, best_target
            start_time = best_t

        # 2. Resample, downmix to mono/stereo, pad to uniform length
        mix_audio = self.preprocessor.process_waveform(
            mix_raw, track.sample_rate, self.target_samples
        )
        target_audio = self.preprocessor.process_waveform(
            target_raw, track.sample_rate, self.target_samples
        )

        # 3. Compute accompaniment residual: r = mix - target
        residual_audio = mix_audio - target_audio

        # 4. Data augmentation (remixing, polarity, rolling)
        if self.augment and self.augmentor is not None:
            mix_audio, target_audio, residual_audio = self.augmentor(
                target_audio, residual_audio
            )

        # 5. Compute time-frequency log-mel spectrograms
        mix_spec, target_spec, res_spec = None, None, None
        if self.return_spectrogram:
            mix_spec = self.preprocessor.compute_spectrogram(mix_audio)
            target_spec = self.preprocessor.compute_spectrogram(target_audio)
            if self.return_residual:
                res_spec = self.preprocessor.compute_spectrogram(residual_audio)

        # 6. Extract acoustic priors (for Style Prior Network)
        priors = {}
        if self.compute_priors and target_spec is not None:
            priors = AcousticPriorExtractor.extract_all(
                target_spec=target_spec,
                stem_type=stem_type,
            )

        sample = SeparationPairSample(
            mix_audio=mix_audio,
            target_audio=target_audio,
            residual_audio=residual_audio if self.return_residual else None,
            mix_spec=mix_spec,
            target_spec=target_spec,
            residual_spec=res_spec,
            stem_type=stem_type,
            stem_name=stem_type.display_name,
            stem_id=int(stem_type),
            sample_rate=self.sample_rate,
            track_id=track.track_id,
            start_time=start_time,
            duration=self.segment_duration,
            priors=priors,
        )
        if self.cache_in_memory:
            self._memory_cache[idx] = sample
        return sample

    def summary(self) -> dict[str, Any]:
        """Provides high-level dataset statistics."""
        return {
            "split": self.split,
            "total_tracks": len(self.tracks),
            "samples_per_track": self.samples_per_track,
            "total_samples": len(self),
            "target_stem": self.target_stem_str,
            "segment_duration_sec": self.segment_duration,
            "energy_threshold": self.energy_threshold,
            "sample_rate": self.sample_rate,
            "channels": 1 if self.mono else 2,
            "first_few_tracks": [t.track_id for t in self.tracks[:5]],
        }


# ---------------------------------------------------------------------------
