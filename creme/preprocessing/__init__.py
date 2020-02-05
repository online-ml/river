"""Feature preprocessing."""
from .feature_hasher import FeatureHasher
from .kernel_approx import RBFSampler
from .one_hot import OneHotEncoder
from .poly import PolynomialExtender
from .scale import Binarizer
from .scale import MaxAbsScaler
from .scale import MinMaxScaler
from .scale import Normalizer
from .scale import RobustScaler
from .scale import StandardScaler


__all__ = [
    'Binarizer',
    'FeatureHasher',
    'MaxAbsScaler',
    'MinMaxScaler',
    'Normalizer',
    'OneHotEncoder',
    'PolynomialExtender',
    'RBFSampler',
    'RobustScaler',
    'StandardScaler'
]
