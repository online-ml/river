"""Multi-armed bandit (MAB) policies.

The bandit policies in River are meant to have a generic API. This allows them to be used in a
variety of contexts. Within River, they are used for model selection
(see `model_selection.BanditRegressor`) as well as recommander systems
(see `reco.BanditRecommender`).

"""

from . import base
from . import envs
from .epsilon_greedy import EpsilonGreedy
from .thompson import ThompsonSampling
from .ucb import UCB

__all__ = ["base", "envs", "EpsilonGreedy", "ThompsonSampling", "UCB"]
