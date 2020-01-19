"""Feature preprocessing."""
from .feature_hasher import FeatureHasher
from .one_hot import OneHotEncoder
from .poly import PolynomialExtender
from .scale import MaxAbsScaler
from .scale import MinMaxScaler
from .scale import Normalizer
from .scale import RobustScaler
from .scale import StandardScaler


__all__ = [
    'FeatureHasher',
    'MaxAbsScaler',
    'MinMaxScaler',
    'Normalizer',
    'OneHotEncoder',
    'PolynomialExtender',
    'RobustScaler',
    'StandardScaler'
]
