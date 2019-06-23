"""
A module for ensemble learning.
"""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .grouping import GroupRegressor
from .hedging import HedgeBinaryClassifier
from .hedging import HedgeRegressor
from .stacking import StackingBinaryClassifier


__all__ = [
    'BaggingClassifier',
    'BaggingRegressor',
    'GroupRegressor',
    'HedgeBinaryClassifier',
    'HedgeRegressor',
    'StackingBinaryClassifier'
]
