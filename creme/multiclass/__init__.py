"""Multi-class classification."""
from .ooc import OutputCodeClassifier
from .ovr import OneVsRestClassifier


__all__ = [
    'OutputCodeClassifier',
    'OneVsRestClassifier'
]
