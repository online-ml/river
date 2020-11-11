"""
The :mod:`skmultiflow.meta` module includes meta learning methods.
"""

from .adaptive_random_forests import AdaptiveRandomForestClassifier
from .adaptive_random_forest_regressor import AdaptiveRandomForestRegressor
from .batch_incremental import BatchIncrementalClassifier
from .leverage_bagging import LeveragingBaggingClassifier
from .oza_bagging import OzaBaggingClassifier
from .oza_bagging_adwin import OzaBaggingADWINClassifier
from .classifier_chains import ClassifierChain
from .classifier_chains import ProbabilisticClassifierChain
from .classifier_chains import MonteCarloClassifierChain
from .regressor_chains import RegressorChain
from .multi_output_learner import MultiOutputLearner
from .learn_pp import LearnPPClassifier
from .learn_nse import LearnPPNSEClassifier
from .accuracy_weighted_ensemble import AccuracyWeightedEnsembleClassifier
from .dynamic_weighted_majority import DynamicWeightedMajorityClassifier
from .additive_expert_ensemble import AdditiveExpertEnsembleClassifier
from .online_boosting import OnlineBoostingClassifier
from .online_adac2 import OnlineAdaC2Classifier
from .online_csb2 import OnlineCSB2Classifier
from .online_under_over_bagging import OnlineUnderOverBaggingClassifier
from .online_rus_boost import OnlineRUSBoostClassifier
from .online_smote_bagging import OnlineSMOTEBaggingClassifier
from .streaming_random_patches import StreamingRandomPatchesClassifier
from .batch_incremental import BatchIncremental   # remove in v0.7.0
from .accuracy_weighted_ensemble import AccuracyWeightedEnsemble   # remove in v0.7.0
from .adaptive_random_forests import AdaptiveRandomForest   # remove in v0.7.0
from .additive_expert_ensemble import AdditiveExpertEnsemble   # remove in v0.7.0
from .dynamic_weighted_majority import DynamicWeightedMajority   # remove in v0.7.0
from .learn_nse import LearnNSE   # remove in v0.7.0
from .learn_pp import LearnPP   # remove in v0.7.0
from .leverage_bagging import LeverageBagging   # remove in v0.7.0
from .online_adac2 import OnlineAdaC2   # remove in v0.7.0
from .online_boosting import OnlineBoosting   # remove in v0.7.0
from .online_csb2 import OnlineCSB2   # remove in v0.7.0
from .online_rus_boost import OnlineRUSBoost   # remove in v0.7.0
from .online_smote_bagging import OnlineSMOTEBagging   # remove in v0.7.0
from .online_under_over_bagging import OnlineUnderOverBagging   # remove in v0.7.0
from .oza_bagging import OzaBagging   # remove in v0.7.0
from .oza_bagging_adwin import OzaBaggingAdwin   # remove in v0.7.0


__all__ = ["AdaptiveRandomForestClassifier", "AdaptiveRandomForestRegressor",
           "BatchIncrementalClassifier", "LeveragingBaggingClassifier", "OzaBaggingClassifier",
           "OzaBaggingADWINClassifier", "ClassifierChain", "ProbabilisticClassifierChain",
           "MonteCarloClassifierChain", "RegressorChain", "MultiOutputLearner",
           "LearnPPClassifier", "LearnPPNSEClassifier", "AccuracyWeightedEnsembleClassifier",
           "DynamicWeightedMajorityClassifier", "AdditiveExpertEnsembleClassifier",
           "OnlineSMOTEBaggingClassifier", "OnlineRUSBoostClassifier", "OnlineCSB2Classifier",
           "OnlineAdaC2Classifier", "OnlineUnderOverBaggingClassifier", "OnlineBoostingClassifier",
           "StreamingRandomPatchesClassifier",
           "BatchIncremental", "AccuracyWeightedEnsemble", "AdaptiveRandomForest",
           "AdditiveExpertEnsemble", "DynamicWeightedMajority", "LearnNSE", "LearnPP",
           "LeverageBagging", "OnlineAdaC2", "OnlineBoosting", "OnlineCSB2", "OnlineRUSBoost",
           "OnlineSMOTEBagging", "OnlineUnderOverBagging", "OzaBagging", "OzaBaggingAdwin"]
