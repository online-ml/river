"""Multi-class classification."""
from .ooc import OutputCodeClassifier
from .ovo import OneVsOneClassifier
from .ovr import OneVsRestClassifier


__all__ = [
    'OutputCodeClassifier',
    'OneVsOneClassifier',
    'OneVsRestClassifier'
]
