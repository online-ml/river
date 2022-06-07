"""Neighbors-based learning.

Also known as *lazy* methods. In these methods, generalisation of the training data is delayed
until a query is received.

"""
from .base import NearestNeighbors
from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor

__all__ = [
    "NearestNeighbors",
    "KNNClassifier",
    "KNNRegressor",
]
