"""
The :mod:`skmultiflow.lazy` module includes lazy learning methods in which generalization of the training
data is delayed until a query is received, this is, on-demand.
"""

from .knn import KNN
from .knn_adwin import KNNAdwin
from .sam_knn import SAMKNN
from .kdtree import KDTree

__all__ = ["KNN", "KNNAdwin", "SAMKNN", "KDTree"]
