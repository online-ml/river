"""
The `river.tree.mondrian` module includes learning and split node
implementations for the Mondrian trees.

Note that this module is not exposed in the tree module, and is instead used by the
AMFClassifier and AMFRegressor classes in the ensemble module.

"""
from __future__ import annotations

from .mondrian_tree import MondrianTree
from .mondrian_tree_classifier import MondrianTreeClassifier
from .mondrian_tree_regressor import MondrianTreeRegressor

__all__ = ["MondrianTree", "MondrianTreeClassifier", "MondrianTreeRegressor"]
