"""Base interfaces.

Every estimator in `creme` is a class, and as such inherits from at least one base interface.
These are used to categorize, organize, and standardize the many estimators that `creme`
contains.

This module contains mixin classes, which are all suffixed by `'Mixin'`. Their purpose is to
provide additional functionality to an estimator, and thus need to be used in conjunction with a
non-mixin base class.

This module also contains utilities for type hinting and tagging estimators.

"""
from . import tags
from . import typing
from .anomaly import AnomalyDetector
from .classifier import Classifier
from .classifier import MiniBatchClassifier
from .clusterer import Clusterer
from .drift_detector import DriftDetector
from .ensemble import EnsembleMixin
from .estimator import Estimator
from .multi_output import MultiOutputClassifier
from .multi_output import MultiOutputRegressor
from .predictor import Predictor
from .regressor import Regressor
from .transformer import SupervisedTransformer
from .transformer import Transformer
from .wrapper import WrapperMixin


__all__ = [
    'AnomalyDetector',
    'BinaryMixin',
    'Classifier',
    'Clusterer',
    'DriftDetector',
    'EnsembleMixin',
    'Estimator',
    'MiniBatchClassifier',
    'MultiClassMixin',
    'MultiOutputClassifier',
    'MultiOutputRegressor',
    'Predictor',
    'Regressor',
    'SupervisedTransformer',
    'tags',
    'Transformer',
    'typing',
    'WrapperMixin'
]
