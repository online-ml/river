"""Compatibility with other libraries.

This module contains wrappers for making ``creme`` estimators compatible with other libraries, and
vice-versa whenever possible.

"""
from .sklearn import convert_creme_to_sklearn
from .sklearn import convert_sklearn_to_creme
from .sklearn import CremeClassifierWrapper
from .sklearn import CremeRegressorWrapper
from .sklearn import SKLRegressorWrapper
from .sklearn import SKLClassifierWrapper
from .sklearn import SKLClustererWrapper
from .sklearn import SKLTransformerWrapper


__all__ = [
    'convert_creme_to_sklearn',
    'convert_sklearn_to_creme',
    'CremeClassifierWrapper',
    'CremeRegressorWrapper',
    'SKLRegressorWrapper',
    'SKLClassifierWrapper',
    'SKLClustererWrapper',
    'SKLTransformerWrapper'
]
