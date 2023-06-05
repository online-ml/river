"""

This module implements incremental Decision Tree (iDT) algorithms for handling classification
and regression tasks.

Each family of iDT will be presented in a dedicated section.

At any moment, iDT might face situations where an input feature previously used to make
a split decision is missing in an incoming sample. In this case, the most traversed path is
selected to pass down the instance. Moreover, in the case of nominal features, if a new category
arises and the feature is used in a decision node, a new branch is created to accommodate the new
value.

**1. Hoeffding Trees**

This family of iDT algorithms use the Hoeffding Bound to determine whether or not the
incrementally computed best split candidates would be equivalent to the ones obtained in a
batch-processing fashion.

All the available Hoeffding Tree (HT) implementation share some common functionalities:

* Set the maximum tree depth allowed (`max_depth`).

* Handle *Active* and *Inactive* nodes: Active learning nodes update their own
internal state to improve predictions and monitor input features to perform split
attempts. Inactive learning nodes do not update their internal state and only keep the
predictors; they are used to save memory in the tree (`max_size`).

*  Enable/disable memory management.

* Define strategies to sort leaves according to how likely they are going to be split.
This enables deactivating non-promising leaves to save memory.

* Disabling ‘poor’ attributes to save memory and speed up tree construction.
A poor attribute is an input feature whose split merit is much smaller than the current
best candidate. Once a feature is disabled, the tree stops saving statistics necessary
to split such a feature.

* Define properties to access leaf prediction strategies, split criteria, and other
relevant characteristics.

**2. Stochastic Gradient Trees**

Stochastic Gradient Trees (SGT) directly optimize a loss function, rather than relying on split
heuristics to guide the tree growth. F-tests are performed do decide whether a leaf should be
expanded or its prediction value should be updated.

SGTs can deal with binary classification and single-target regression. They also support
dynamic and static feature quantizers to deal with numerical inputs.

"""
from __future__ import annotations

from . import base, splitter
from .extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from .hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier
from .hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .isoup_tree_regressor import iSOUPTreeRegressor
from .stochastic_gradient_tree import SGTClassifier, SGTRegressor

__all__ = [
    "base",
    "splitter",
    "ExtremelyFastDecisionTreeClassifier",
    "HoeffdingTreeClassifier",
    "HoeffdingAdaptiveTreeClassifier",
    "HoeffdingTreeRegressor",
    "HoeffdingAdaptiveTreeRegressor",
    "iSOUPTreeRegressor",
    "SGTClassifier",
    "SGTRegressor",
]
