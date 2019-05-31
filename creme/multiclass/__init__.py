"""
Meta-estimators for performing multi-class classification with binary classifiers.
"""
from .ovo import OneVsOneClassifier
from .ovr import OneVsRestClassifier


__all__ = [
    'OneVsOneClassifier',
    'OneVsRestClassifier'
]
