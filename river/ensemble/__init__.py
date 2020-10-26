"""Ensemble learning."""
from .bagging import ADWINBaggingClassifier
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .bagging import LeveragingBaggingClassifier
from .boosting import AdaBoostClassifier
from .streaming_random_patches import StreamingRandomPatchesClassifier


__all__ = [
    'AdaBoostClassifier',
    'ADWINBaggingClassifier',
    'BaggingClassifier',
    'BaggingRegressor',
    'LeveragingBaggingClassifier',
    'StreamingRandomPatchesClassifier'
]
