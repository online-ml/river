"""
Module for preparing data to make it compatible with certain machine learning algorithms.
"""
from .feature_hasher import FeatureHasher
from .one_hot import OneHotEncoder
from .scale import StandardScaler


__all__ = [
    'FeatureHasher',
    'OneHotEncoder',
    'StandardScaler'
]
