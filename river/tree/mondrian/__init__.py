"""
The `river.tree.mondrian` module includes learning and split node
implementations for the Mondrian trees.

Note that this module is not exposed in the tree module, and is instead used by the
AMFClassifier class in the ensemble module.

"""

from .mondrian_tree import MondrianTree
from .mondrian_tree_classifier import MondrianTreeClassifier

__all__ = ["MondrianTree", "MondrianTreeClassifier"]
