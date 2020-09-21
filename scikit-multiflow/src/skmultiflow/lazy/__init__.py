"""
The :mod:`skmultiflow.lazy` module includes lazy learning methods in which generalization of the training
data is delayed until a query is received, this is, on-demand.
"""

from .knn_classifier import KNNClassifier
from .knn_adwin import KNNADWINClassifier
# from .sam_knn import SAMKNNClassifier
from .knn_classifier import KNN   # remove in v0.7.0
from .knn_adwin import KNNAdwin   # remove in v0.7.0
# from .sam_knn import SAMKNN   # remove in v0.7.0
from .knn_regressor import KNNRegressor


__all__ = ["KNNClassifier", "KNNADWINClassifier", # "SAMKNNClassifier",
           "KNNRegressor", "KNN", "KNNAdwin", "SAMKNN"]
