"""Expert learning.

This module regroups a variety of methods that may be used for performing model selection. An
expert learner is provided with a list of models, which are also called experts, and is tasked with
performing at least as well as the best expert. Indeed, initially the best model is not known. The
performance of each model becomes more apparent as time goes by. Different strategies are possible,
each one offering a different tradeoff in terms of accuracy and computational performance.

Expert learning can be used for tuning the hyperparameters of a model. This may be done by creating
a copy of the model for each set of hyperparameters, and treating each copy as a separate model.
The `utils.expand_param_grid` function can be used for this purpose.

Note that this differs from the `ensemble` module in that methods from the latter are designed to
improve the performance of a single model. Both modules may thus be used in conjunction with one
another.

"""

from .bandit import EpsilonGreedyRegressor, UCBRegressor
from .ewa import EWARegressor
from .sh import SuccessiveHalvingClassifier, SuccessiveHalvingRegressor
from .stacking import StackingClassifier

__all__ = [
    "EpsilonGreedyRegressor",
    "EWARegressor",
    "SuccessiveHalvingClassifier",
    "SuccessiveHalvingRegressor",
    "StackingClassifier",
    "UCBRegressor",
]
