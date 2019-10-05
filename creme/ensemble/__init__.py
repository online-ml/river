"""Ensemble learning."""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .hedging import HedgeRegressor
from .stacking import StackingBinaryClassifier


__all__ = [
    'BaggingClassifier',
    'BaggingRegressor',
    'HedgeRegressor',
    'StackingBinaryClassifier'
]
