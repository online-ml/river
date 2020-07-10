"""
The :mod:`skmultiflow.core` module covers core elements of `scikit-multiflow`.
"""

from .base import BaseSKMObject
from .base import ClassifierMixin
from .base import RegressorMixin
from .base import MetaEstimatorMixin
from .base import MultiOutputMixin
from .base import clone
from .base import is_classifier
from .base import is_regressor
from .pipeline import Pipeline

__all__ = ["BaseSKMObject", "ClassifierMixin", "RegressorMixin", "MetaEstimatorMixin",
           "MultiOutputMixin", "Pipeline", "clone", "is_classifier", "is_regressor"]
