"""Neighbors-based learning."""
from .knn_adwin import KNNADWINClassifier
from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor
from .sam_knn import SAMKNNClassifier


__all__ = [
    'KNNADWINClassifier',
    'KNNClassifier',
    'KNNRegressor',
    'SAMKNNClassifier'
]
