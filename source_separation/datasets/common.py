"""
Single Source Separation & Style Prior Dataset Module
======================================================
This module provides a unified data loading, preprocessing, augmentation,
and prior extraction pipeline for training Single Source Separation models
and Audio Style Prior Networks (AudioSPN).

Supported Datasets:
  - MUSDB18 / MUSDB18-HQ (using multi-stream .stem.mp4 or directory stems)
  - Slakh2100 / MoisesDB / MedleyDB / DSD100 (using paired audio files)
  - Custom paired audio directories (mix.wav <-> <instrument>.wav)

Taxonomy of Stems (MUSDB18 standard):
  0: Mixture     - Full composite musical mix
  1: Drums       - Percussion, kicks, snares, cymbals, transients
  2: Bass        - Bass guitar, synth bass, low tonal drone
  3: Other       - Guitars, keyboards, synthesizers, orchestral accompaniment
  4: Vocals      - Lead and backing vocals, speech, melodic voice
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

# Set up module logger
logger = logging.getLogger("source_separation_dataset")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 1. Stem Taxonomy & Enumeration
# ---------------------------------------------------------------------------


class StemType(IntEnum):
    """Standard stem taxonomy based on MUSDB18 multitrack stem specifications."""

    MIXTURE = 0
    DRUMS = 1
    BASS = 2
    OTHER = 3
    VOCALS = 4

    @classmethod
    def from_str(cls, name: str | StemType) -> StemType:
        """Parse stem identifier from string with alias normalization."""
        if isinstance(name, cls):
            return name
        name_clean = name.strip().lower()
        mapping = {
            # Mixture
            "mixture": cls.MIXTURE,
            "mix": cls.MIXTURE,
            "song": cls.MIXTURE,
            "original": cls.MIXTURE,
            # Drums
            "drums": cls.DRUMS,
            "drum": cls.DRUMS,
            "percussion": cls.DRUMS,
            "beat": cls.DRUMS,
            "tabla": cls.DRUMS,
            "mridangam": cls.DRUMS,
            # Bass
            "bass": cls.BASS,
            "drone": cls.BASS,
            "tanpura": cls.BASS,
            "sub": cls.BASS,
            # Other
            "other": cls.OTHER,
            "accompaniment": cls.OTHER,
            "acc": cls.OTHER,
            "inst": cls.OTHER,
            "keys": cls.OTHER,
            "guitar": cls.OTHER,
            "strings": cls.OTHER,
            # Vocals
            "vocals": cls.VOCALS,
            "vocal": cls.VOCALS,
            "voice": cls.VOCALS,
            "lead": cls.VOCALS,
            "singing": cls.VOCALS,
        }
        if name_clean in mapping:
            return mapping[name_clean]
        raise ValueError(
            f"Unknown stem name '{name}'. Valid options: {list(mapping.keys())}"
        )

    @property
    def display_name(self) -> str:
        return self.name.lower()

    @property
    def is_instrument(self) -> bool:
        return self != StemType.MIXTURE

    @property
    def prior_focus(self) -> str:
        """Dominant acoustic prior feature characteristic for this stem."""
        focus_map = {
            StemType.MIXTURE: "composite_timbre",
            StemType.DRUMS: "rhythm_transients",
            StemType.BASS: "harmonic_chroma",
            StemType.OTHER: "timbral_texture",
            StemType.VOCALS: "melodic_contour",
        }
        return focus_map.get(self, "acoustic_style")


# ---------------------------------------------------------------------------
# 2. Structured Data Containers
# ---------------------------------------------------------------------------


@dataclass
class SeparationPairSample:
    """Structured representation of a single source separation training pair.

    Contains waveform tensors, time-frequency spectrogram representations,
    extracted acoustic prior features, and full track metadata.
    """

    # Time-domain waveforms [channels, samples] in range [-1.0, 1.0]
    mix_audio: Tensor
    target_audio: Tensor
    residual_audio: Tensor | None = None

    # Time-frequency representations [channels, n_mels/freq, frames]
    mix_spec: Tensor | None = None
    target_spec: Tensor | None = None
    residual_spec: Tensor | None = None

    # Stem identification
    stem_type: StemType = StemType.DRUMS
    stem_name: str = "drums"
    stem_id: int = 1

    # Audio physical parameters
    sample_rate: int = 44100
    track_id: str = "unknown"
    start_time: float = 0.0
    duration: float = 10.0

    # Acoustic priors extracted for Style Prior Network conditioning
    priors: dict[str, Tensor] | None = field(default_factory=dict)

    def to(self, device: torch.device) -> SeparationPairSample:
        """Transfer all PyTorch tensors in sample to target compute device."""
        return SeparationPairSample(
            mix_audio=self.mix_audio.to(device),
            target_audio=self.target_audio.to(device),
            residual_audio=(
                self.residual_audio.to(device)
                if self.residual_audio is not None
                else None
            ),
            mix_spec=(
                self.mix_spec.to(device) if self.mix_spec is not None else None
            ),
            target_spec=(
                self.target_spec.to(device)
                if self.target_spec is not None
                else None
            ),
            residual_spec=(
                self.residual_spec.to(device)
                if self.residual_spec is not None
                else None
            ),
            stem_type=self.stem_type,
            stem_name=self.stem_name,
            stem_id=self.stem_id,
            sample_rate=self.sample_rate,
            track_id=self.track_id,
            start_time=self.start_time,
            duration=self.duration,
            priors=(
                {k: v.to(device) for k, v in self.priors.items()}
                if self.priors
                else {}
            ),
        )


@dataclass
class TrackMetadata:
    """Metadata describing a track in a multitrack dataset."""

    track_id: str
    file_path: Path
    artist: str = ""
    title: str = ""
    duration: float = 0.0
    sample_rate: int = 44100
    channels: int = 2
    stem_activity: dict[str, list[float]] = field(default_factory=dict)


def find_active_time_offsets(
    target_audio: Tensor,
    needed_samples: int,
    sample_rate: int,
    energy_threshold: float = 0.005,
    min_active_ratio: float = 0.5,
    hop_sec: float = 0.5,
) -> list[int]:
    """Finds non-silent sample start offsets for target stem in waveform tensor.

    Excludes silent segments and guarantees that the target stem is active
    over the sliced segment duration.

    Args:
        target_audio: Waveform tensor [channels, samples].
        needed_samples: Total samples needed for the segment (e.g. 10s * sr).
        sample_rate: Audio sampling rate in Hz.
        energy_threshold: Minimum RMS threshold to qualify as non-silent.
        min_active_ratio: Minimum fraction of sub-frames that must be active.
        hop_sec: Stride in seconds between candidate window evaluations.

    Returns:
        List of sample indices suitable for starting a non-silent slice.
    """
    total_samples = target_audio.shape[-1]
    max_start = total_samples - needed_samples
    if max_start <= 0:
        return [0]

    hop_samples = max(1, int(hop_sec * sample_rate))
    sub_frame_samples = max(1, int(0.5 * sample_rate))

    # Average over channels -> [1, 1, total_samples]
    sq = target_audio.pow(2).mean(dim=0, keepdim=True).unsqueeze(0)

    # 1. Window RMS
    win_energies = torch.nn.functional.avg_pool1d(
        sq, kernel_size=needed_samples, stride=hop_samples
    ).squeeze()
    if win_energies.ndim == 0:
        win_energies = win_energies.unsqueeze(0)
    win_rms = torch.sqrt(win_energies + 1e-8)

    # 2. Sub-frame RMS to verify sustained instrument activity vs transient noise
    sub_energies = torch.nn.functional.avg_pool1d(
        sq, kernel_size=sub_frame_samples, stride=sub_frame_samples
    ).squeeze()
    if sub_energies.ndim == 0:
        sub_energies = sub_energies.unsqueeze(0)
    sub_rms = torch.sqrt(sub_energies + 1e-8)
    sub_active = (sub_rms >= energy_threshold).float()

    frames_per_win = max(1, needed_samples // sub_frame_samples)
    hop_frames = max(1, hop_samples // sub_frame_samples)
    act_ratio = torch.nn.functional.avg_pool1d(
        sub_active.view(1, 1, -1),
        kernel_size=frames_per_win,
        stride=hop_frames,
    ).squeeze()
    if act_ratio.ndim == 0:
        act_ratio = act_ratio.unsqueeze(0)

    num_cand = min(len(win_rms), len(act_ratio))
    win_rms = win_rms[:num_cand]
    act_ratio = act_ratio[:num_cand]

    mask = (win_rms >= energy_threshold) & (act_ratio >= min_active_ratio)
    qualified = torch.where(mask)[0]
    if len(qualified) == 0:
        qualified = torch.where(win_rms >= energy_threshold)[0]
    if len(qualified) == 0:
        qualified = torch.tensor([torch.argmax(win_rms).item()])

    return [min(int(i.item()) * hop_samples, max_start) for i in qualified]


# ---------------------------------------------------------------------------
# 3. Acoustic Prior Extractor
# ---------------------------------------------------------------------------


class AcousticPriorExtractor:
    """Extracts structured musical and acoustic style priors from audio stems.

    These priors correspond to the conditioning signals required by:
      - Notebook 06: Audio Style Prior Network (AudioSPN)
      - Notebook 09: Source Separation Priors (Rhythm, Harmonic, Melodic)
      - Notebook 10: Stem-Conditioned Latent Diffusion Steering
    """

    @staticmethod
    def extract_rhythm_prior(spec_or_mel: Tensor) -> Tensor:
        """Extracts onset spectral flux and temporal transients (rhythm envelope).

        Args:
            spec_or_mel: (C, F, T) or (1, F, T) spectrogram.
        Returns:
            (C, T) normalized onset transient energy curve in [0, 1].
        """
        if spec_or_mel.ndim == 2:
            spec_or_mel = spec_or_mel.unsqueeze(0)
        # First-order positive temporal difference (spectral flux)
        diff = torch.diff(
            spec_or_mel, dim=-1, prepend=spec_or_mel[:, :, :1]
        )
        onset_flux = torch.relu(diff).sum(dim=1)  # (C, T)
        # Normalize per-channel
        norm_factor = onset_flux.amax(dim=-1, keepdim=True) + 1e-6
        return onset_flux / norm_factor

    @staticmethod
    def extract_harmonic_prior(
        spec_or_mel: Tensor, num_chroma: int = 12
    ) -> Tensor:
        """Extracts a 12-bin musical chromagram from harmonic stems (e.g. Bass/Drone).

        Args:
            spec_or_mel: (C, F, T) or (F, T) spectrogram.
            num_chroma: Number of pitch classes (default 12 for semitones).
        Returns:
            (num_chroma, T) normalized chroma energy distribution.
        """
        if spec_or_mel.ndim == 3:
            # Average across channels if stereo
            spec_2d = spec_or_mel.mean(dim=0)  # (F, T)
        else:
            spec_2d = spec_or_mel

        F_bins, T_frames = spec_2d.shape
        device = spec_2d.device

        # Deterministic logarithmic frequency-to-chroma projection
        chroma_weights = torch.zeros(
            (num_chroma, F_bins), dtype=spec_2d.dtype, device=device
        )
        for f in range(F_bins):
            pitch_class = int(math.log2(f + 1.0) * 12.0) % num_chroma
            chroma_weights[pitch_class, f] = 1.0

        chroma = chroma_weights @ spec_2d  # (12, T)
        norm = chroma.sum(dim=0, keepdim=True) + 1e-6
        return chroma / norm

    @staticmethod
    def extract_melodic_prior(spec_or_mel: Tensor) -> Tensor:
        """Extracts spectral centroid and dominant pitch trajectory from lead stems.

        Args:
            spec_or_mel: (C, F, T) spectrogram.
        Returns:
            (C, T) normalized spectral centroid contour in [0, 1].
        """
        if spec_or_mel.ndim == 2:
            spec_or_mel = spec_or_mel.unsqueeze(0)
        C, F_bins, T_frames = spec_or_mel.shape
        device = spec_or_mel.device

        freq_axis = torch.arange(
            F_bins, dtype=spec_or_mel.dtype, device=device
        ).view(1, F_bins, 1)
        weighted_freq = (spec_or_mel * freq_axis).sum(dim=1)  # (C, T)
        energy = spec_or_mel.sum(dim=1) + 1e-6
        centroid = weighted_freq / energy  # (C, T)
        return centroid / float(max(F_bins - 1, 1))

    @staticmethod
    def extract_energy_envelope(spec_or_mel: Tensor) -> Tensor:
        """Extracts temporal root-mean-square (RMS) energy envelope.

        Args:
            spec_or_mel: (C, F, T) spectrogram.
        Returns:
            (C, T) normalized energy envelope curve.
        """
        if spec_or_mel.ndim == 2:
            spec_or_mel = spec_or_mel.unsqueeze(0)
        rms = torch.sqrt(spec_or_mel.pow(2).mean(dim=1) + 1e-6)
        norm = rms.amax(dim=-1, keepdim=True) + 1e-6
        return rms / norm

    @classmethod
    def extract_all(
        cls,
        target_spec: Tensor,
        stem_type: StemType = StemType.DRUMS,
    ) -> dict[str, Tensor]:
        """Extracts complete prior suite formatted for Style Prior Networks."""
        priors = {
            "rhythm": cls.extract_rhythm_prior(target_spec),
            "envelope": cls.extract_energy_envelope(target_spec),
            "melodic": cls.extract_melodic_prior(target_spec),
            "harmonic": cls.extract_harmonic_prior(target_spec),
        }
        return priors


# ---------------------------------------------------------------------------
# 4. Audio Preprocessor & Spectrogram Transform
# ---------------------------------------------------------------------------


class AudioPreprocessor:
    """Handles audio resampling, channel downmixing/upmixing, length alignment,
    and log-mel spectrogram computation matching the AudioSPN architecture.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        mono: bool = True,
        normalize_mode: str = "peak",
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float | None = None,
        patch_frames: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize_mode = normalize_mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or (sample_rate / 2.0)
        self.patch_frames = patch_frames

        # Log-Mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )

    def process_waveform(
        self,
        waveform: Tensor,
        orig_sr: int,
        target_samples: int | None = None,
    ) -> Tensor:
        """Resamples, downmixes to mono/stereo, normalizes, and pads/crops waveform."""
        # Ensure 2D tensor [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.t()

        # Resample if sample rates differ
        if orig_sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_sr, self.sample_rate
            )

        # Channel downmix / upmix
        if self.mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif not self.mono and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Pad or crop to exact sample length if specified
        if target_samples is not None:
            curr_len = waveform.shape[-1]
            if curr_len < target_samples:
                pad_amt = target_samples - curr_len
                waveform = torch.nn.functional.pad(
                    waveform, (0, pad_amt), mode="constant", value=0.0
                )
            elif curr_len > target_samples:
                waveform = waveform[..., :target_samples]

        # Normalization
        return self.normalize(waveform)

    def normalize(self, waveform: Tensor) -> Tensor:
        """Normalizes waveform according to normalize_mode ('peak' or 'rms')."""
        if self.normalize_mode == "peak":
            peak = waveform.abs().max()
            if peak > 1e-4:
                return (waveform / peak * 0.95).float()
        elif self.normalize_mode == "rms":
            rms = torch.sqrt(torch.mean(waveform**2) + 1e-8)
            return (waveform / rms * 0.1).float()
        return waveform.float()

    def compute_spectrogram(self, waveform: Tensor) -> Tensor:
        """Computes normalized log-mel spectrogram [channels, n_mels, frames]."""
        # mel: [channels, n_mels, frames]
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))

        # Align to fixed patch frames if requested (e.g. for AudioSPN training)
        if self.patch_frames is not None:
            curr_frames = log_mel.shape[-1]
            if curr_frames < self.patch_frames:
                log_mel = torch.nn.functional.pad(
                    log_mel,
                    (0, self.patch_frames - curr_frames),
                    mode="replicate",
                )
            elif curr_frames > self.patch_frames:
                log_mel = log_mel[..., : self.patch_frames]

        return log_mel


# ---------------------------------------------------------------------------
# 5. Audio Augmentation Suite
# ---------------------------------------------------------------------------


class SeparationAugmentor:
    """Specialized music source separation and style prior augmentation suite.

    Features:
      - Dynamic Stem Remixing: adjusts target stem volume relative to background mix
        and recombines mix = target * g_tgt + residual * g_res.
      - Polarity Inversion: randomly negates waveforms (phase flip).
      - Channel Swapping: stereo left/right swap.
      - Temporal Cyclic Roll: time shifts audio without boundary cuts.
      - Subtle Gaussian Noise Injection.
    """

    def __init__(
        self,
        gain_perturb_db: float = 4.0,
        prob_polarity_flip: float = 0.5,
        prob_channel_swap: float = 0.5,
        prob_time_roll: float = 0.3,
        prob_noise: float = 0.2,
        noise_snr_db: float = 30.0,
    ) -> None:
        self.gain_perturb_db = gain_perturb_db
        self.prob_polarity_flip = prob_polarity_flip
        self.prob_channel_swap = prob_channel_swap
        self.prob_time_roll = prob_time_roll
        self.prob_noise = prob_noise
        self.noise_snr_db = noise_snr_db

    def __call__(
        self,
        target_audio: Tensor,
        residual_audio: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Applies coordinated augmentations to target and residual, then synthesizes augmented mix.

        Returns:
            Tuple: (augmented_mix, augmented_target, augmented_residual)
        """
        # 1. Dynamic Remixing: Perturb relative stem gains
        if self.gain_perturb_db > 0:
            target_gain_db = random.uniform(
                -self.gain_perturb_db, self.gain_perturb_db
            )
            res_gain_db = random.uniform(
                -self.gain_perturb_db, self.gain_perturb_db
            )
            g_tgt = 10.0 ** (target_gain_db / 20.0)
            g_res = 10.0 ** (res_gain_db / 20.0)
            target_audio = target_audio * g_tgt
            residual_audio = residual_audio * g_res

        # 2. Phase / Polarity Inversion
        if random.random() < self.prob_polarity_flip:
            target_audio = -target_audio
            residual_audio = -residual_audio

        # 3. Stereo Channel Swap
        if (
            target_audio.shape[0] == 2
            and random.random() < self.prob_channel_swap
        ):
            target_audio = target_audio.flip(0)
            residual_audio = residual_audio.flip(0)

        # 4. Temporal Roll
        if random.random() < self.prob_time_roll:
            shift = random.randint(0, target_audio.shape[-1] - 1)
            target_audio = torch.roll(target_audio, shifts=shift, dims=-1)
            residual_audio = torch.roll(residual_audio, shifts=shift, dims=-1)

        # 5. Composite Mix Reconstruction (guaranteeing exact energy sum)
        mix_audio = target_audio + residual_audio

        # 6. Additive subtle white noise injection
        if random.random() < self.prob_noise:
            noise = torch.randn_like(mix_audio)
            signal_power = mix_audio.pow(2).mean() + 1e-8
            noise_power = noise.pow(2).mean() + 1e-8
            desired_ratio = 10.0 ** (-self.noise_snr_db / 10.0)
            scale = torch.sqrt(desired_ratio * signal_power / noise_power)
            mix_audio = mix_audio + noise * scale

        # Prevent digital clipping
        peak = mix_audio.abs().max()
        if peak > 0.99:
            scale = 0.95 / peak
            mix_audio = mix_audio * scale
            target_audio = target_audio * scale
            residual_audio = residual_audio * scale

        return mix_audio, target_audio, residual_audio


# ---------------------------------------------------------------------------
# 8. Directory-Based Paired Dataset (Slakh, MoisesDB, MedleyDB, Custom WAVs)
# ---------------------------------------------------------------------------


class DirectorySingleSourceDataset(Dataset):
    """Universal paired dataset supporting open-source datasets distributed as audio files.

    Supported dataset layouts:
      1) Track Subdirectories (e.g. Slakh2100, MUSDB18-HQ, MoisesDB):
         root_dir/
           Track0001/
             mix.wav (or mix.flac)
             drums.wav (or bass.wav, vocals.wav, etc.)
      2) Parallel Folders:
         root_dir/
           mix/
             song01.wav
           drums/
             song01.wav
    """

    def __init__(
        self,
        root_dir: str | Path,
        target_stem: str | StemType = "drums",
        segment_duration: float = 10.0,
        sample_rate: int = 44100,
        mono: bool = True,
        return_spectrogram: bool = True,
        return_residual: bool = True,
        compute_priors: bool = True,
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
        self.root_dir = Path(root_dir)
        self.target_stem = StemType.from_str(target_stem)
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

        self.track_pairs = self._discover_pairs()
        if not self.track_pairs:
            raise FileNotFoundError(
                f"No matching (mix, {self.target_stem.display_name}) audio pairs found in {self.root_dir}."
            )

        logger.info(
            f"Initialized DirectorySingleSourceDataset: found {len(self.track_pairs)} paired tracks "
            f"for stem '{self.target_stem.display_name}'. Total samples: {len(self)} (Duration: {self.segment_duration}s)."
        )

    def _discover_pairs(self) -> list[dict[str, Any]]:
        """Scans directory hierarchy for matching mixture and target stem files."""
        pairs = []
        stem_name = self.target_stem.display_name
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}

        # Pattern 1: Folder per track
        for track_dir in sorted(self.root_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            mix_file = None
            target_file = None

            for f in track_dir.iterdir():
                if f.suffix.lower() not in audio_extensions:
                    continue
                name_stem = f.stem.lower()
                if name_stem in {"mix", "mixture", "composite"}:
                    mix_file = f
                elif (
                    stem_name in name_stem
                    or name_stem == stem_name
                    or f"{stem_name}.stem" in name_stem
                ):
                    target_file = f

            if mix_file is not None and target_file is not None:
                pairs.append(
                    {
                        "track_id": track_dir.name,
                        "mix_path": mix_file,
                        "target_path": target_file,
                    }
                )

        # Pattern 2: Parallel folders (e.g. root/mix/*.wav and root/drums/*.wav)
        if not pairs:
            mix_dir = self.root_dir / "mix"
            target_dir = self.root_dir / stem_name
            if mix_dir.exists() and target_dir.exists():
                for mix_file in sorted(mix_dir.iterdir()):
                    if mix_file.suffix.lower() not in audio_extensions:
                        continue
                    tgt_file = target_dir / mix_file.name
                    if tgt_file.exists():
                        pairs.append(
                            {
                                "track_id": mix_file.stem,
                                "mix_path": mix_file,
                                "target_path": tgt_file,
                            }
                        )

        return pairs

    def __len__(self) -> int:
        return len(self.track_pairs) * self.samples_per_track

    def __getitem__(self, idx: int) -> SeparationPairSample:
        pair_idx = idx // self.samples_per_track
        pair = self.track_pairs[pair_idx]

        # Read audio files using soundfile
        mix_data, mix_sr = sf.read(str(pair["mix_path"]), dtype="float32")
        tgt_data, tgt_sr = sf.read(str(pair["target_path"]), dtype="float32")

        # Convert to torch tensor [channels, samples]
        mix_t = torch.from_numpy(mix_data).t()
        tgt_t = torch.from_numpy(tgt_data).t()
        if mix_t.ndim == 1:
            mix_t = mix_t.unsqueeze(0)
        if tgt_t.ndim == 1:
            tgt_t = tgt_t.unsqueeze(0)

        # Slicing time segment
        min_samples = min(mix_t.shape[-1], tgt_t.shape[-1])
        needed_samples = int(self.segment_duration * mix_sr)

        if min_samples > needed_samples:
            cand_offsets = find_active_time_offsets(
                target_audio=tgt_t[..., :min_samples],
                needed_samples=needed_samples,
                sample_rate=mix_sr,
                energy_threshold=self.energy_threshold,
                min_active_ratio=self.min_active_ratio,
            )
            if self.random_crop:
                start_samp = random.choice(cand_offsets)
            else:
                sub_idx = idx % self.samples_per_track
                step_idx = min(
                    int(sub_idx * len(cand_offsets) / self.samples_per_track),
                    len(cand_offsets) - 1,
                )
                start_samp = cand_offsets[step_idx]

            mix_crop = mix_t[..., start_samp : start_samp + needed_samples]
            tgt_crop = tgt_t[..., start_samp : start_samp + needed_samples]
            start_sec = start_samp / mix_sr
        else:
            mix_crop = mix_t[..., :min_samples]
            tgt_crop = tgt_t[..., :min_samples]
            start_sec = 0.0

        # Preprocess
        mix_audio = self.preprocessor.process_waveform(
            mix_crop, mix_sr, self.target_samples
        )
        target_audio = self.preprocessor.process_waveform(
            tgt_crop, tgt_sr, self.target_samples
        )
        residual_audio = mix_audio - target_audio

        # Augmentation
        if self.augment and self.augmentor is not None:
            mix_audio, target_audio, residual_audio = self.augmentor(
                target_audio, residual_audio
            )

        # Spectrograms
        mix_spec, target_spec, res_spec = None, None, None
        if self.return_spectrogram:
            mix_spec = self.preprocessor.compute_spectrogram(mix_audio)
            target_spec = self.preprocessor.compute_spectrogram(target_audio)
            if self.return_residual:
                res_spec = self.preprocessor.compute_spectrogram(residual_audio)

        # Priors
        priors = {}
        if self.compute_priors and target_spec is not None:
            priors = AcousticPriorExtractor.extract_all(
                target_spec=target_spec,
                stem_type=self.target_stem,
            )

        return SeparationPairSample(
            mix_audio=mix_audio,
            target_audio=target_audio,
            residual_audio=residual_audio if self.return_residual else None,
            mix_spec=mix_spec,
            target_spec=target_spec,
            residual_spec=res_spec,
            stem_type=self.target_stem,
            stem_name=self.target_stem.display_name,
            stem_id=int(self.target_stem),
            sample_rate=self.sample_rate,
            track_id=pair["track_id"],
            start_time=start_sec,
            duration=self.segment_duration,
            priors=priors,
        )


# ---------------------------------------------------------------------------
# 9. PyTorch DataLoader Batch Collator
# ---------------------------------------------------------------------------


class SourceSeparationCollator:
    """Collates individual `SeparationPairSample` objects into batched PyTorch tensors."""

    def __call__(
        self, samples: list[SeparationPairSample]
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {}

        # 1. Batched waveforms: [B, channels, samples]
        batch["mix_audio"] = torch.stack([s.mix_audio for s in samples], dim=0)
        batch["target_audio"] = torch.stack(
            [s.target_audio for s in samples], dim=0
        )

        if samples[0].residual_audio is not None:
            batch["residual_audio"] = torch.stack(
                [s.residual_audio for s in samples if s.residual_audio is not None],
                dim=0,
            )

        # 2. Batched spectrograms: [B, channels, n_mels, frames]
        if samples[0].mix_spec is not None:
            batch["mix_spec"] = torch.stack(
                [s.mix_spec for s in samples if s.mix_spec is not None], dim=0
            )
        if samples[0].target_spec is not None:
            batch["target_spec"] = torch.stack(
                [s.target_spec for s in samples if s.target_spec is not None],
                dim=0,
            )
        if samples[0].residual_spec is not None:
            batch["residual_spec"] = torch.stack(
                [s.residual_spec for s in samples if s.residual_spec is not None],
                dim=0,
            )

        # 3. Categorical & Metadata
        batch["stem_ids"] = torch.tensor(
            [s.stem_id for s in samples], dtype=torch.long
        )
        batch["stem_names"] = [s.stem_name for s in samples]
        batch["track_ids"] = [s.track_id for s in samples]
        batch["start_times"] = torch.tensor(
            [s.start_time for s in samples], dtype=torch.float32
        )
        batch["durations"] = torch.tensor(
            [s.duration for s in samples], dtype=torch.float32
        )
        batch["sample_rates"] = torch.tensor(
            [s.sample_rate for s in samples], dtype=torch.long
        )

        # 4. Batched Acoustic Priors (Rhythm, Chroma, Centroid, Envelope)
        if samples[0].priors:
            priors_batch: dict[str, Tensor] = {}
            prior_keys = samples[0].priors.keys()
            for k in prior_keys:
                tensors = [s.priors[k] for s in samples if k in s.priors]
                if len(tensors) == len(samples):
                    priors_batch[k] = torch.stack(tensors, dim=0)
            batch["priors"] = priors_batch

        return batch


# ---------------------------------------------------------------------------
# 10. Convenience Factory Functions
# ---------------------------------------------------------------------------
