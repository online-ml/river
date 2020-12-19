"""Tree-based methods.

This modules contains tree-based methods. Similar to the batch learning versions, these methods
model data by means of a tree-structure. However, these methods create the tree incrementally.

"""
from ._base_tree import BaseHoeffdingTree
from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier
from .extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from .label_combination_hoeffding_tree import LabelCombinationHoeffdingTreeClassifier
from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from .isoup_tree_regressor import iSOUPTreeRegressor


__all__ = [
    "BaseHoeffdingTree",
    "HoeffdingTreeClassifier",
    "HoeffdingAdaptiveTreeClassifier",
    "ExtremelyFastDecisionTreeClassifier",
    "LabelCombinationHoeffdingTreeClassifier",
    "HoeffdingTreeRegressor",
    "HoeffdingAdaptiveTreeRegressor",
    "iSOUPTreeRegressor",
]
