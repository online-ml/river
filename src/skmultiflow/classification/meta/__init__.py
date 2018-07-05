"""
The :mod:`skmultiflow.classification.meta` module includes meta learning methods.
"""

from .adaptive_random_forests import AdaptiveRandomForest
from .batch_incremental import BatchIncremental
from .leverage_bagging import LeverageBagging
from .oza_bagging import OzaBagging
from .oza_bagging_adwin import OzaBaggingAdwin

__all__ = ["AdaptiveRandomForest", "BatchIncremental", "LeverageBagging", "OzaBagging", "OzaBaggingAdwin"]
