"""
The :mod:`skmultiflow.trees` module includes learning methods based on trees.
"""
from . import attribute_observer
from . import attribute_test
from . import nodes
from . import split_criterion
from .hoeffding_tree import HoeffdingTree
from .hoeffding_adaptive_tree import HAT
from .hoeffding_anytime_tree import HATT
from .lc_hoeffding_tree import LCHT
from .regression_hoeffding_tree import RegressionHoeffdingTree
from .regression_hoeffding_adaptive_tree import RegressionHAT
from .multi_target_regression_hoeffding_tree import MultiTargetRegressionHoeffdingTree
from .stacked_single_target_hoeffding_tree_regressor import StackedSingleTargetHoeffdingTreeRegressor

__all__ = ["attribute_observer", "attribute_test", "nodes", "split_criterion",
           "HoeffdingTree", "HAT", "HATT", "LCHT", "RegressionHoeffdingTree",
           "RegressionHAT", "MultiTargetRegressionHoeffdingTree",
           "StackedSingleTargetHoeffdingTreeRegressor"]

