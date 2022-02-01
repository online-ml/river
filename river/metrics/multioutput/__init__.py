"""Metrics for multi-output learning."""
from .base import MultiOutputClassificationMetric, MultiOutputRegressionMetric
from .confusion import MultiLabelConfusionMatrix
from .exact_match import ExactMatch
from .micro import MicroAverage
from .per_output import PerOutput


__all__ = [
    "MultiLabelConfusionMatrix",
    "MultiOutputClassificationMetric",
    "MultiOutputRegressionMetric",
    "ExactMatch",
    "MicroAverage",
    "PerOutput",
]
