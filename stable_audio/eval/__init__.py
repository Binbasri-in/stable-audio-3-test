"""Evaluation module for audio generation and conditioned audio generation."""

from .metrics import (
    calculate_frechet_distance,
    compute_fad,
    compute_clap_score,
    compute_inpaint_metrics,
    BenchmarkTimer,
)
from .dataset import (
    SDDPair,
    SongDescriberDataset,
    is_instrumental_and_coherent,
)
from .embeddings import (
    CLAPFeatureExtractor,
    EmbeddedCLAPEncoder,
)

__all__ = [
    "calculate_frechet_distance",
    "compute_fad",
    "compute_clap_score",
    "compute_inpaint_metrics",
    "BenchmarkTimer",
    "SDDPair",
    "SongDescriberDataset",
    "is_instrumental_and_coherent",
    "CLAPFeatureExtractor",
    "EmbeddedCLAPEncoder",
]
