"""
The :mod:`skmultiflow.trees` module includes learning methods based on trees.
"""
from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .hoeffding_adaptive_tree import HoeffdingAdaptiveTreeClassifier
from .extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from .label_combination_hoeffding_tree import LabelCombinationHoeffdingTreeClassifier
from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from .isoup_tree_regressor import iSOUPTreeRegressor

__all__ = ["HoeffdingTreeClassifier", "HoeffdingAdaptiveTreeClassifier",
           "ExtremelyFastDecisionTreeClassifier", "LabelCombinationHoeffdingTreeClassifier",
           "HoeffdingTreeRegressor", "HoeffdingAdaptiveTreeRegressor", "iSOUPTreeRegressor"]
