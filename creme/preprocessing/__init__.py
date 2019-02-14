"""
Module for preparing data to make it compatible with certain machine learning algorithms.
"""
from .feature_hasher import FeatureHasher
from .one_hot import OneHotEncoder
from .poly import PolynomialExtender
from .scale import StandardScaler
from .scale import MinMaxScaler


__all__ = [
    'FeatureHasher',
    'MinMaxScaler',
    'OneHotEncoder',
    'PolynomialExtender',
    'StandardScaler'
]
