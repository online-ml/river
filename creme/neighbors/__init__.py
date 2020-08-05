"""Neighbors-based learning."""
from .knn import KNeighborsClassifier
from .knn import KNeighborsRegressor
from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor


__all__ = [
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    'KNNClassifier',
    'KNNRegressor'
]
