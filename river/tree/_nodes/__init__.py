"""
The `river.tree._nodes` module includes learning and split node
implementations for the hoeffding trees.
"""

# Base class nodes
from .base import FoundNode
from .base import Node
from .base import SplitNode
from .base import LearningNode
from .base import ActiveLeaf
from .base import InactiveLeaf
# Hoeffding Tree classifier nodes
from .htc_nodes import ActiveLearningNodeMC
from .htc_nodes import InactiveLearningNodeMC
from .htc_nodes import ActiveLearningNodeNB
from .htc_nodes import ActiveLearningNodeNBA
# Hoeffding Adaptive Tree classifier nodes
from .hatc_nodes import AdaNode
from .hatc_nodes import AdaSplitNodeClassifier
from .hatc_nodes import AdaLearningNodeClassifier
# Adaptive Random Forest classifier nodes
from .arf_htc_nodes import RandomActiveLearningNodeMC
from .arf_htc_nodes import RandomActiveLearningNodeNB
from .arf_htc_nodes import RandomActiveLearningNodeNBA
# Extremely Fast Decision Tree classifier nodes
from .efdtc_nodes import EFDTSplitNode
from .efdtc_nodes import EFDTActiveLearningNodeMC
from .efdtc_nodes import EFDTInactiveLearningNodeMC
from .efdtc_nodes import EFDTActiveLearningNodeNB
from .efdtc_nodes import EFDTActiveLearningNodeNBA
# Hoeffding Tree regressor nodes
from .htr_nodes import ActiveLearningNodeMean
from .htr_nodes import ActiveLearningNodeModel
from .htr_nodes import ActiveLearningNodeAdaptive
from .htr_nodes import InactiveLearningNodeMean
from .htr_nodes import InactiveLearningNodeModel
from .htr_nodes import InactiveLearningNodeAdaptive
# Hoeffding Adaptive Tree regressor nodes
from .hatr_nodes import AdaSplitNodeRegressor
from .hatr_nodes import AdaActiveLearningNodeRegressor
# Adaptive Random Forest regressor nodes
from .arf_htr_nodes import RandomActiveLearningNodeMean
from .arf_htr_nodes import RandomActiveLearningNodeModel
from .arf_htr_nodes import RandomActiveLearningNodeAdaptive
# iSOUP-Tree regressor nodes
from .isouptr_nodes import ActiveLearningNodeModelMultiTarget
from .isouptr_nodes import ActiveLearningNodeAdaptiveMultiTarget
from .isouptr_nodes import InactiveLearningNodeModelMultiTarget
from .isouptr_nodes import InactiveLearningNodeAdaptiveMultiTarget


__all__ = [
    'FoundNode',
    'Node',
    'SplitNode',
    'LearningNode',
    'ActiveLeaf',
    'InactiveLeaf',
    'AdaNode',
    'ActiveLearningNodeMC',
    'InactiveLearningNodeMC',
    'ActiveLearningNodeNB',
    'ActiveLearningNodeNBA',
    'RandomActiveLearningNodeMC',
    'RandomActiveLearningNodeNB',
    'RandomActiveLearningNodeNBA',
    'AdaSplitNodeClassifier',
    'AdaLearningNodeClassifier',
    'EFDTSplitNode',
    'EFDTActiveLearningNodeMC',
    'EFDTInactiveLearningNodeMC',
    'EFDTActiveLearningNodeNB',
    'EFDTActiveLearningNodeNBA',
    'ActiveLearningNodeMean',
    'ActiveLearningNodeModel',
    'ActiveLearningNodeAdaptive',
    'InactiveLearningNodeMean',
    'InactiveLearningNodeModel',
    'InactiveLearningNodeAdaptive',
    'RandomActiveLearningNodeMean',
    'RandomActiveLearningNodeModel',
    'RandomActiveLearningNodeAdaptive',
    'AdaSplitNodeRegressor',
    'AdaActiveLearningNodeRegressor',
    'ActiveLearningNodeModelMultiTarget',
    'ActiveLearningNodeAdaptiveMultiTarget',
    'InactiveLearningNodeModelMultiTarget',
    'InactiveLearningNodeAdaptiveMultiTarget'
]
