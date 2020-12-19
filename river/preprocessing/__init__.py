"""Feature preprocessing.

The purpose of this module is to modify an existing set of features so that they can be processed
by a machine learning algorithm. This may be done by scaling numeric parts of the data or by
one-hot encoding categorical features. The difference with the `feature_extraction` module is that
the latter extracts new information from the data

"""
from .feature_hasher import FeatureHasher
from .impute import PreviousImputer
from .impute import StatImputer
from .lda import LDA
from .one_hot import OneHotEncoder
from .scale import Binarizer
from .scale import MaxAbsScaler
from .scale import MinMaxScaler
from .scale import Normalizer
from .scale import RobustScaler
from .scale import StandardScaler


__all__ = [
    "Binarizer",
    "FeatureHasher",
    "LDA",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "OneHotEncoder",
    "PreviousImputer",
    "RobustScaler",
    "StandardScaler",
    "StatImputer",
]
