"""Model selection.

This module regroups a variety of methods that may be used for performing model selection. An
model selector is provided with a list of models. These are called "experts" in the expert learning
litterature. The model selector's goal is to perform at least as well as the best model. Indeed,
initially, the best model is not known. The performance of each model becomes more apparent as time
goes by. Different strategies are possible, each one offering a different tradeoff in terms of
accuracy and computational performance.

Broadly speaking, there are two kinds of model selection approaches:

1. Those that look for the best model. The best performing model in the past is the one that gets
    to make a prediction in the future. This is the realm of bandits.
2. Those that trust each model according to its performance. Each model contributes to a prediction
    according to how accurate it was in the past. This is realm of expert learning.

Both approaches are into a single module to provide a common interface.

Model selection can be used for tuning the hyperparameters of a model. This may be done by creating
a copy of the model for each set of hyperparameters, and treating each copy as a separate model.
The `utils.expand_param_grid` function can be used for this purpose.

Note that this differs from the `ensemble` module in that methods from the latter are designed to
improve the performance of a single model. Indeed, in an ensemble, multiple copies of the same are
made. The model outputs are usually aggregated with a weighted scheme. Both modules are thus
complementary to one another.

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
