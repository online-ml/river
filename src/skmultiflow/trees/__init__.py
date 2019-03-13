"""
The :mod:`skmultiflow.trees` module includes learning methods based on trees.
"""

from .hoeffding_tree import HoeffdingTree
from .hoeffding_adaptive_tree import HAT
from .hoeffding_anytime_tree import HATT
from .lc_hoeffding_tree import LCHT
from .regression_hoeffding_tree import RegressionHoeffdingTree
from .regression_hoeffding_adaptive_tree import RegressionHAT
from .multi_target_regression_hoeffding_tree import MultiTargetRegressionHoeffdingTree

__all__ = ["HoeffdingTree", "HAT", "HATT", "LCHT", "RegressionHoeffdingTree", "RegressionHAT",
           "MultiTargetRegressionHoeffdingTree"]
