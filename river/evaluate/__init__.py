"""Model evaluation.

This module provides utilities to evaluate an online model. The goal is to reproduce a real-world
scenario with high fidelity. The core function of this module is `progressive_val_score`, which
allows to evaluate a model via progressive validation.

This module also exposes "tracks". A track is a predefined combination of a dataset and one or more
metrics. This allows a principled manner to compare models with each other. For instance,
the `RegressionTrack` contains several datasets and metrics to evaluate regression models. There is
also a bare `Track` class to implement a custom track. The `benchmarks` directory at the root of
the River repository uses these tracks.

"""
from __future__ import annotations

from .progressive_validation import iter_progressive_val_score, progressive_val_score
from .tracks import BinaryClassificationTrack, MultiClassClassificationTrack, RegressionTrack, Track

__all__ = [
    "iter_progressive_val_score",
    "progressive_val_score",
    "BinaryClassificationTrack",
    "MultiClassClassificationTrack",
    "RegressionTrack",
    "Track",
]
