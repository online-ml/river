"""Model selection.

This module regroups a variety of methods that may be used for performing model selection. An
model selector is provided with a list of models. These are called "experts" in the expert learning
literature. The model selector's goal is to perform at least as well as the best model. Indeed,
initially, the best model is not known. The performance of each model becomes more apparent as time
goes by. Different strategies are possible, each one offering a different tradeoff in terms of
accuracy and computational performance.

Model selection can be used for tuning the hyperparameters of a model. This may be done by creating
a copy of the model for each set of hyperparameters, and treating each copy as a separate model.
The `utils.expand_param_grid` function can be used for this purpose.

"""

from . import base
from .epsilon_greedy import EpsilonGreedyRegressor
from .greedy import GreedyRegressor
from .sh import SuccessiveHalvingClassifier, SuccessiveHalvingRegressor
from .ucb import UCBRegressor

__all__ = [
    "base",
    "EpsilonGreedyRegressor",
    "GreedyRegressor",
    "SuccessiveHalvingClassifier",
    "SuccessiveHalvingRegressor",
    "UCBRegressor",
]
