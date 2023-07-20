"""Metrics for multi-output learning."""
from __future__ import annotations

from . import base
from .confusion import MultiLabelConfusionMatrix
from .exact_match import ExactMatch
from .macro import MacroAverage
from .micro import MicroAverage
from .per_output import PerOutput
from .accuracy_moa_jaccard import Accuracy_MOA_Jaccard

__all__ = [
    "base",
    "MacroAverage",
    "MultiLabelConfusionMatrix",
    "ExactMatch",
    "MicroAverage",
    "PerOutput",
    "Accuracy_MOA_Jaccard",
]
