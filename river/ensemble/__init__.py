"""Ensemble learning."""
from .bagging import ADWINBaggingClassifier
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .boosting import AdaBoostClassifier


__all__ = [
    'AdaBoostClassifier',
    'ADWINBaggingClassifier',
    'BaggingClassifier',
    'BaggingRegressor',
]
