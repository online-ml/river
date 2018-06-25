"""
The :mod:`skmultiflow.evaluation` module includes evaluation methods for stream learning.
"""

from .evaluate_prequential import EvaluatePrequential
from .evaluate_holdout import EvaluateHoldout

__all__ = ["EvaluatePrequential", "EvaluateHoldout"]
