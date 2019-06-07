"""
Module for preparing data to make it compatible with certain machine learning algorithms.
"""
from .feature_hasher import FeatureHasher
from .one_hot import OneHotEncoder
from .poly import PolynomialExtender
from .scale import MinMaxScaler
from .scale import Normalizer
from .scale import StandardScaler


__all__ = [
    'FeatureHasher',
    'MinMaxScaler',
    'Normalizer',
    'OneHotEncoder',
    'PolynomialExtender',
    'StandardScaler'
]
