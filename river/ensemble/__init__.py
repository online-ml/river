"""Ensemble learning."""
from .adaptive_random_forest import AdaptiveRandomForestClassifier
from .bagging import ADWINBaggingClassifier
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .bagging import LeveragingBaggingClassifier
from .boosting import AdaBoostClassifier


__all__ = [
    'AdaptiveRandomForestClassifier',
    'AdaBoostClassifier',
    'ADWINBaggingClassifier',
    'BaggingClassifier',
    'BaggingRegressor',
    'LeveragingBaggingClassifier'
]
