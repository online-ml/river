"""
A module for ensemble learning.
"""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .hedge import HedgeClassifier


__all__ = [
    'BaggingClassifier',
    'BaggingRegressor',
    'HedgeClassifier'
]
