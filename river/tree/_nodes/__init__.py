"""
The `river.tree._nodes` module includes learning and split node
implementations for the hoeffding trees.
"""

# Base class nodes
from .base import FoundNode
from .base import Node
from .base import SplitNode
from .base import LearningNode

# Hoeffding Tree classifier nodes
from .htc_nodes import LearningNodeMC
from .htc_nodes import LearningNodeNB
from .htc_nodes import LearningNodeNBA

# Hoeffding Adaptive Tree classifier nodes
from .hatc_nodes import AdaNode
from .hatc_nodes import AdaSplitNodeClassifier
from .hatc_nodes import AdaLearningNodeClassifier

# Adaptive Random Forest classifier nodes
from .arf_htc_nodes import RandomLearningNodeMC
from .arf_htc_nodes import RandomLearningNodeNB
from .arf_htc_nodes import RandomLearningNodeNBA

# Extremely Fast Decision Tree classifier nodes
from .efdtc_nodes import EFDTSplitNode
from .efdtc_nodes import EFDTLearningNodeMC
from .efdtc_nodes import EFDTLearningNodeNB
from .efdtc_nodes import EFDTLearningNodeNBA

# Hoeffding Tree regressor nodes
from .htr_nodes import LearningNodeMean
from .htr_nodes import LearningNodeModel
from .htr_nodes import LearningNodeAdaptive

# Hoeffding Adaptive Tree regressor nodes
from .hatr_nodes import AdaSplitNodeRegressor
from .hatr_nodes import AdaLearningNodeRegressor

# Adaptive Random Forest regressor nodes
from .arf_htr_nodes import RandomLearningNodeMean
from .arf_htr_nodes import RandomLearningNodeModel
from .arf_htr_nodes import RandomLearningNodeAdaptive

# iSOUP-Tree regressor nodes
from .isouptr_nodes import LearningNodeMeanMultiTarget
from .isouptr_nodes import LearningNodeModelMultiTarget
from .isouptr_nodes import LearningNodeAdaptiveMultiTarget


__all__ = [
    "FoundNode",
    "Node",
    "SplitNode",
    "LearningNode",
    "AdaNode",
    "LearningNodeMC",
    "LearningNodeNB",
    "LearningNodeNBA",
    "RandomLearningNodeMC",
    "RandomLearningNodeNB",
    "RandomLearningNodeNBA",
    "AdaSplitNodeClassifier",
    "AdaLearningNodeClassifier",
    "EFDTSplitNode",
    "EFDTLearningNodeMC",
    "EFDTLearningNodeNB",
    "EFDTLearningNodeNBA",
    "LearningNodeMean",
    "LearningNodeModel",
    "LearningNodeAdaptive",
    "LearningNodeMean",
    "LearningNodeModel",
    "LearningNodeAdaptive",
    "RandomLearningNodeMean",
    "RandomLearningNodeModel",
    "RandomLearningNodeAdaptive",
    "AdaSplitNodeRegressor",
    "AdaLearningNodeRegressor",
    "LearningNodeMeanMultiTarget",
    "LearningNodeModelMultiTarget",
    "LearningNodeAdaptiveMultiTarget",
]
