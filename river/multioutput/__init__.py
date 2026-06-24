"""Multi-output models."""

from __future__ import annotations

from .chain import (
    ClassifierChain,
    MonteCarloClassifierChain,
    ProbabilisticClassifierChain,
    RegressorChain,
)
from .encoder import MultiClassEncoder
from .per_output import PerOutputRegressor
from .per_output_classifier import PerOutputClassifier

__all__ = [
    "ClassifierChain",
    "MonteCarloClassifierChain",
    "MultiClassEncoder",
    "PerOutputClassifier",
    "PerOutputRegressor",
    "ProbabilisticClassifierChain",
    "RegressorChain",
]
