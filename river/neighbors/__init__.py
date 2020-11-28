"""Neighbors-based learning.

Also known as *lazy* methods. In these methods, generalisation of the training data is delayed
until a query is received.

"""
from .knn_adwin import KNNADWINClassifier
from .knn_classifier import KNNClassifier
from .knn_regressor import KNNRegressor
from .sam_knn import SAMKNNClassifier


__all__ = ["KNNADWINClassifier", "KNNClassifier", "KNNRegressor", "SAMKNNClassifier"]
