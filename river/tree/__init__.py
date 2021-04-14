"""

This module implements incremental Decision Tree (iDT) algorithms for handling classification
and regression tasks.

Each family of iDT will be presented in a dedicated section.

At any moment, iDT might face situations where an input feature previously used to make
a split decision is missing in an incoming sample. In this case, the river's trees follow the
conventions:

- *Learning:* choose the subtree branch most traversed so far to pass the instance on.</br>
    * In case of nominal features, a new branch is created to accommodate the new
    category.</br>
- *Predicting:* Use the last "reachable" decision node to provide responses.

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

All HTs have the following parameters, in addition to their own, that can be selected
using `**kwargs`. The following default values are used, unless otherwise explicitly stated
in the tree documentation.

| Parameter | Description | Default |
| :- | :- | -: |
|`max_depth` | The maximum depth a tree can reach. If `None`, the tree will grow indefinitely. | `None` |
| `binary_split` | If True, only allow binary splits. | `False` |
| `max_size` | The maximum size the tree can reach, in Megabytes (MB). | `100` |
| `memory_estimate_period` | Interval (number of processed instances) between memory consumption checks. | `1_000_000` |
| `stop_mem_management` | If True, stop growing as soon as memory limit is hit. | `False` |
| `remove_poor_attrs` | If True, disable poorly descriptive attributes to reduce memory usage. | `False` |
| `merit_preprune` | If True, enable merit-based tree pre-pruning. | `True` |

"""

from . import splitter
from .extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from .hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier
from .hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from .hoeffding_tree_classifier import HoeffdingTreeClassifier
from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .isoup_tree_regressor import iSOUPTreeRegressor
from .label_combination_hoeffding_tree import LabelCombinationHoeffdingTreeClassifier

__all__ = [
    "splitter",
    "HoeffdingTreeClassifier",
    "ExtremelyFastDecisionTreeClassifier",
    "HoeffdingAdaptiveTreeClassifier",
    "HoeffdingTreeRegressor",
    "HoeffdingAdaptiveTreeRegressor",
    "iSOUPTreeRegressor",
    "LabelCombinationHoeffdingTreeClassifier",
]
