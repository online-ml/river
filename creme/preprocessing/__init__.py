"""
Module for preparing data to make it compatible with certain machine learning algorithms.
"""
from .discard import Discarder
from .feature_hasher import FeatureHasher
from .func import FuncTransformer
from .one_hot import OneHotEncoder
from .poly import PolynomialExtender
from .scale import StandardScaler
from .scale import MinMaxScaler


__all__ = [
    'Discarder',
    'FeatureHasher',
    'FuncTransformer',
    'MinMaxScaler',
    'OneHotEncoder',
    'PolynomialExtender',
    'StandardScaler'
]
