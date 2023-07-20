"""Metrics for multi-output learning."""
from __future__ import annotations

from . import base
from .confusion import MultiLabelConfusionMatrix
from .exact_match import ExactMatch
from .macro import MacroAverage
from .micro import MicroAverage
from .per_output import PerOutput
from .sample_average import SampleAverage

__all__ = [
    "base",
    "MacroAverage",
    "MultiLabelConfusionMatrix",
    "ExactMatch",
    "MicroAverage",
    "PerOutput",
    "SampleAverage",
]
