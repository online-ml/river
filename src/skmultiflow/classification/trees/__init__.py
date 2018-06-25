"""
The :mod:`skmultiflow.classification.trees` module includes learning methods based on trees.
"""

from .hoeffding_tree import HoeffdingTree
from .hoeffding_adaptive_tree import HAT

__all__ = ["HoeffdingTree", "HAT"]
