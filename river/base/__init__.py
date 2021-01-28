"""Base interfaces.

Every estimator in `river` is a class, and as such inherits from at least one base interface.
These are used to categorize, organize, and standardize the many estimators that `river`
contains.

This module contains mixin classes, which are all suffixed by `Mixin`. Their purpose is to
provide additional functionality to an estimator, and thus need to be used in conjunction with a
non-mixin base class.

This module also contains utilities for type hinting and tagging estimators.

"""
from . import tags, typing
from .anomaly import AnomalyDetector
from .base import Base
from .classifier import Classifier, MiniBatchClassifier
from .clusterer import Clusterer
from .drift_detector import DriftDetector
from .ensemble import EnsembleMixin
from .estimator import Estimator
from .multi_output import MultiOutputMixin
from .regressor import MiniBatchRegressor, Regressor
from .transformer import SupervisedTransformer, Transformer
from .wrapper import WrapperMixin

__all__ = [
    "AnomalyDetector",
    "Base",
    "Classifier",
    "Clusterer",
    "DriftDetector",
    "EnsembleMixin",
    "Estimator",
    "MiniBatchClassifier",
    "MiniBatchRegressor",
    "MultiOutputMixin",
    "Regressor",
    "SupervisedTransformer",
    "tags",
    "Transformer",
    "typing",
    "WrapperMixin",
]
