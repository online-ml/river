"""Ensemble learning."""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .boosting import AdaBoostClassifier
from .hedging import HedgeRegressor
from .stacking import StackingBinaryClassifier


__all__ = [
    'AdaBoostClassifier',
    'BaggingClassifier',
    'BaggingRegressor',
    'HedgeRegressor',
    'StackingBinaryClassifier',
]
