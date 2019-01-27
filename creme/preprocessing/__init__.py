"""
Module for preparing data to make it compatible with certain machine learning algorithms.
information
"""
from .hash import FeatureHasher
from .one_hot import OneHotEncoder
from .scale import StandardScaler


__all__ = [
    'FeatureHasher',
    'OneHotEncoder',
    'StandardScaler'
]
