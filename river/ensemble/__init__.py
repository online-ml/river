"""Ensemble learning.

Broadly speaking, there are two kinds of ensemble approaches:

1. Those that copy a model multiple and aggregate the predictions of said copies. This encompasses
    bagging as well as boosting.
2. Those that take as input an arbitrary list of models.

"""
from .adaptive_random_forest import (
    AdaptiveRandomForestClassifier,
    AdaptiveRandomForestRegressor,
)
from .bagging import (
    ADWINBaggingClassifier,
    BaggingClassifier,
    BaggingRegressor,
    LeveragingBaggingClassifier,
)
from .boosting import AdaBoostClassifier
from .ewa import EWARegressor
from .stacking import StackingClassifier
from .streaming_random_patches import SRPClassifier, SRPRegressor
from .voting import VotingClassifier

__all__ = [
    "AdaptiveRandomForestClassifier",
    "AdaptiveRandomForestRegressor",
    "AdaBoostClassifier",
    "ADWINBaggingClassifier",
    "BaggingClassifier",
    "BaggingRegressor",
    "EWARegressor",
    "LeveragingBaggingClassifier",
    "SRPClassifier",
    "SRPRegressor",
    "StackingClassifier",
    "VotingClassifier",
]
