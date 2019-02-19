"""
A module for ensemble learning.
"""
from .bagging import BaggingClassifier
from .hedge import HedgeClassifier


__all__ = ['BaggingClassifier', 'HedgeClassifier']
