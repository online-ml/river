"""Feature preprocessing.

The purpose of this module is to modify an existing set of features so that they can be processed
by a machine learning algorithm. This may be done by scaling numeric parts of the data or by
one-hot encoding categorical features. The difference with the `feature_extraction` module is that
the latter extracts new information from the data

"""
from __future__ import annotations

from .feature_hasher import FeatureHasher
from .impute import PreviousImputer, StatImputer
from .lda import LDA
from .one_hot import OneHotEncoder
from .ordinal import OrdinalEncoder
from .pred_clipper import PredClipper
from .random_projection import GaussianRandomProjector, SparseRandomProjector
from .scale import (
    AdaptiveStandardScaler,
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from .scale_target import TargetMinMaxScaler, TargetStandardScaler

__all__ = [
    "AdaptiveStandardScaler",
    "Binarizer",
    "FeatureHasher",
    "GaussianRandomProjector",
    "LDA",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PredClipper",
    "PreviousImputer",
    "RobustScaler",
    "SparseRandomProjector",
    "StandardScaler",
    "StatImputer",
    "TargetMinMaxScaler",
    "TargetStandardScaler",
]
