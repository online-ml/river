"""
The `river.tree.mondrian` module includes learning and split node
implementations for the Mondrian trees.
"""

from .mondrian_tree import MondrianTree
from .mondrian_tree_classifier import MondrianTreeClassifier

__all__ = ["MondrianTree", "MondrianTreeClassifier"]
