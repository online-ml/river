"""Ensemble learning."""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .boosting import AdaBoostClassifier


__all__ = [
    'AdaBoostClassifier',
    'BaggingClassifier',
    'BaggingRegressor',
]
