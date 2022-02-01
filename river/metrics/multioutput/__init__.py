"""Metrics for multi-output learning."""
from .base import MultiOutputClassificationMetric, MultiOutputRegressionMetric
from .confusion import MultiLabelConfusionMatrix
from .exact_match import ExactMatch
from .macro import MacroAverage
from .micro import MicroAverage
from .per_output import PerOutput

__all__ = [
    "MacroAverage",
    "MultiLabelConfusionMatrix",
    "MultiOutputClassificationMetric",
    "MultiOutputRegressionMetric",
    "ExactMatch",
    "MicroAverage",
    "PerOutput",
]
