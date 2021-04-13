"""
The `river.tree._nodes` module includes learning and split node
implementations for the hoeffding trees.
"""

from .arf_htc_nodes import (
    RandomLeafMajorityClass,
    RandomLeafNaiveBayes,
    RandomLeafNaiveBayesAdaptive,
)
from .arf_htr_nodes import RandomLeafAdaptive, RandomLeafMean, RandomLeafModel
from .branch import (
    HTBranch,
    NominalBinaryBranch,
    NominalMultiwayBranch,
    NumericBinaryBranch,
    NumericMultiwayBranch,
)
from .efdtc_nodes import (
    BaseEFDTBranch,
    BaseEFDTLeaf,
    EFDTLeafMajorityClass,
    EFDTLeafNaiveBayes,
    EFDTLeafNaiveBayesAdaptive,
    EFDTNominalBinaryBranch,
    EFDTNominalMultiwayBranch,
    EFDTNumericBinaryBranch,
    EFDTNumericMultiwayBranch,
)
from .hatc_nodes import (
    AdaBranchClassifier,
    AdaLeafClassifier,
    AdaNode,
    AdaNomBinaryBranchClass,
    AdaNomMultiwayBranchClass,
    AdaNumBinaryBranchClass,
    AdaNumMultiwayBranchClass,
)

# from .hatr_nodes import AdaLearningNodeRegressor, AdaSplitNodeRegressor
from .htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from .htr_nodes import LeafAdaptive, LeafMean, LeafModel
from .isouptr_nodes import (
    LeafAdaptiveMultiTarget,
    LeafMeanMultiTarget,
    LeafModelMultiTarget,
)
from .leaf import HTLeaf

__all__ = [
    "HTBranch",
    "HTLeaf",
    "NominalBinaryBranch",
    "NominalMultiwayBranch",
    "NumericBinaryBranch",
    "NumericMultiwayBranch",
    "AdaNode",
    "LeafMajorityClass",
    "LeafNaiveBayes",
    "LeafNaiveBayesAdaptive",
    "RandomLeafMajorityClass",
    "RandomLeafNaiveBayes",
    "RandomLeafNaiveBayesAdaptive",
    "AdaBranchClassifier",
    "AdaNomBinaryBranchClass",
    "AdaNomMultiwayBranchClass",
    "AdaNumBinaryBranchClass",
    "AdaNumMultiwayBranchClass",
    "AdaLeafClassifier",
    "BaseEFDTBranch",
    "BaseEFDTLeaf",
    "EFDTNominalBinaryBranch",
    "EFDTNominalMultiwayBranch",
    "EFDTNumericBinaryBranch",
    "EFDTNumericMultiwayBranch",
    "EFDTLeafMajorityClass",
    "EFDTLeafNaiveBayes",
    "EFDTLeafNaiveBayesAdaptive",
    "LeafMean",
    "LeafModel",
    "LeafAdaptive",
    "LeafMean",
    "LeafModel",
    "LeafAdaptive",
    "RandomLeafMean",
    "RandomLeafModel",
    "RandomLeafAdaptive",
    # "AdaSplitNodeRegressor",
    # "AdaLearningNodeRegressor",
    "LeafMeanMultiTarget",
    "LeafModelMultiTarget",
    "LeafAdaptiveMultiTarget",
]
