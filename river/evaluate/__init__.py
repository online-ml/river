"""Model evaluation.

This module provides utilities to evaluate an online model. The goal is to reproduce a real-world
scenario with high fidelity. The core function of this module is `progressive_val_score`, which
allows to evaluate a model via progressive validation.

This module also exposes "tracks". A track is a predefined combination of a dataset and one or more
metrics. This allows a principled manner to compare models with each other. For instance,
the `load_binary_clf_tracks` returns tracks that are to be used to evaluate the performance of
a binary classification model.

The `benchmarks` directory at the root of the River repository uses these tracks.

"""
from .progressive_validation import progressive_val_score
from .tracks import Track, load_binary_clf_tracks

__all__ = ["load_binary_clf_tracks", "progressive_val_score", "Track"]
