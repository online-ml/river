"""
The :mod:`skmultiflow.meta` module includes meta learning methods.
"""

from .adaptive_random_forests import AdaptiveRandomForest
from .batch_incremental import BatchIncremental
from .leverage_bagging import LeverageBagging
from .oza_bagging import OzaBagging
from .oza_bagging_adwin import OzaBaggingAdwin
from .classifier_chains import ClassifierChain
from .classifier_chains import ProbabilisticClassifierChain
from .classifier_chains import MonteCarloClassifierChain
from .regressor_chains import RegressorChain
from .multi_output_learner import MultiOutputLearner
from .learn_pp import LearnPP
from .learn_nse import LearnNSE
from .accuracy_weighted_ensemble import AccuracyWeightedEnsemble
from .dynamic_weighted_majority import DynamicWeightedMajority
from .additive_expert_ensemble import AdditiveExpertEnsemble
from .online_boosting import OnlineBoosting
from .online_adac2 import OnlineAdaC2
from .online_csb2 import OnlineCSB2
from .online_under_over_bagging import OnlineUnderOverBagging
from .online_rus_boost import OnlineRUSBoost
from .online_smote_bagging import OnlineSMOTEBagging


__all__ = ["AdaptiveRandomForest", "BatchIncremental", "LeverageBagging", "OzaBagging", "OzaBaggingAdwin",
           "ClassifierChain", "ProbabilisticClassifierChain", "MonteCarloClassifierChain",
           "RegressorChain", "MultiOutputLearner", "LearnPP", "LearnNSE", "AccuracyWeightedEnsemble",
           "DynamicWeightedMajority", "AdditiveExpertEnsemble", "OnlineSMOTEBagging",
           "OnlineRUSBoost", "OnlineCSB2", "OnlineAdaC2", "OnlineUnderOverBagging", "OnlineBoosting"]
