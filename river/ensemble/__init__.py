"""Ensemble learning.

This module includes ensemble methods. This kind of methods improve predictive performance by
combining the prediction of their members.

"""
from .adaptive_random_forest import AdaptiveRandomForestClassifier
from .adaptive_random_forest import AdaptiveRandomForestRegressor
from .bagging import ADWINBaggingClassifier
from .bagging import BaggingClassifier
from .bagging import BaggingRegressor
from .bagging import LeveragingBaggingClassifier
from .boosting import AdaBoostClassifier
from .streaming_random_patches import SRPClassifier


__all__ = [
    "AdaptiveRandomForestClassifier",
    "AdaptiveRandomForestRegressor",
    "AdaBoostClassifier",
    "ADWINBaggingClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
    "LeveragingBaggingClassifier",
    "SRPClassifier",
]
