"""Neighbors-based learning.

Also known as *lazy* methods. In these methods, generalisation of the training data is delayed
until a query is received.

"""
from .base_neighbors import BaseKNN
from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor
from .neighbors import MinkowskiNeighbors, NearestNeighbors

__all__ = [
    "BaseKNN",
    "MinkowskiNeighbors",
    "NearestNeighbors",
    "KNNClassifier",
    "KNNRegressor",
]
