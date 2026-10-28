"""Convenience DataLoader factories for source separation datasets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .datasets.common import SourceSeparationCollator, StemType, logger
from .datasets.moisesdb import MoisesDBSingleSourceDataset
from .datasets.musdb18 import MUSDB18SingleSourceDataset
from .datasets.slakh2100 import SlakhSingleSourceDataset


def create_source_separation_dataloader(
    root_dir: str | Path | None = None,
    split: str = "train",
    target_stem: str | StemType = "drums",
    batch_size: int = 16,
    num_workers: int = 0,
    segment_duration: float = 10.0,
    sample_rate: int = 44100,
    mono: bool = True,
    augment: bool = False,
    shuffle: bool = True,
    samples_per_track: int = 10,
    energy_threshold: float = 0.005,
    min_active_ratio: float = 0.5,
    patch_frames: int | None = None,
    max_tracks: int | None = None,
    cache_in_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for MUSDB18 single-source separation."""
    dataset = MUSDB18SingleSourceDataset(
        root_dir=root_dir, split=split, target_stem=target_stem,
        segment_duration=segment_duration, sample_rate=sample_rate, mono=mono,
        return_spectrogram=True, return_residual=True, compute_priors=True,
        augment=augment, samples_per_track=samples_per_track,
        energy_threshold=energy_threshold, min_active_ratio=min_active_ratio,
        patch_frames=patch_frames, max_tracks=max_tracks,
        cache_in_memory=cache_in_memory,
    )
    logger.info(dataset.summary())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=SourceSeparationCollator(), pin_memory=torch.cuda.is_available(),
    )


def create_moisesdb_dataloader(
    root_dir: str | Path | None = None,
    target_stem: str | StemType = "drums",
    batch_size: int = 16,
    num_workers: int = 0,
    segment_duration: float = 10.0,
    sample_rate: int = 44100,
    mono: bool = True,
    augment: bool = False,
    shuffle: bool = True,
    samples_per_track: int = 5,
    energy_threshold: float = 0.005,
    min_active_ratio: float = 0.5,
) -> DataLoader:
    """Create a DataLoader for MoisesDB single-source separation."""
    dataset = MoisesDBSingleSourceDataset(
        root_dir=root_dir, target_stem=target_stem, segment_duration=segment_duration,
        sample_rate=sample_rate, mono=mono, return_spectrogram=False,
        return_residual=True, compute_priors=False, augment=augment,
        samples_per_track=samples_per_track, energy_threshold=energy_threshold,
        min_active_ratio=min_active_ratio,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=SourceSeparationCollator(), pin_memory=torch.cuda.is_available(),
    )


def create_slakh_dataloader(
    root_dir: str | Path | None = None,
    target_stem: str | StemType = "guitar",
    batch_size: int = 16,
    num_workers: int = 0,
    segment_duration: float = 10.0,
    sample_rate: int = 44100,
    mono: bool = True,
    augment: bool = False,
    shuffle: bool = True,
    samples_per_track: int = 5,
    energy_threshold: float = 0.005,
    min_active_ratio: float = 0.5,
) -> DataLoader:
    """Create a DataLoader for Slakh2100 single-source separation."""
    dataset = SlakhSingleSourceDataset(
        root_dir=root_dir, target_stem=target_stem, segment_duration=segment_duration,
        sample_rate=sample_rate, mono=mono, return_spectrogram=False,
        return_residual=True, compute_priors=False, augment=augment,
        samples_per_track=samples_per_track, energy_threshold=energy_threshold,
        min_active_ratio=min_active_ratio,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=SourceSeparationCollator(), pin_memory=torch.cuda.is_available(),
    )
