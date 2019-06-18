"""
A module for ensemble learning.
"""
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .group import GroupRegressor
from .hedge import HedgeClassifier
from .hedge import HedgeRegressor


__all__ = [
    'BaggingClassifier',
    'BaggingRegressor',
    'GroupRegressor',
    'HedgeClassifier',
    'HedgeRegressor'
]
