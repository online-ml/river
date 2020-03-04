"""Base interfaces.

Every estimator in ``creme`` is a class, and as such inherits from at least one base interface.
These are used to categorize, organize, and standardize the many estimators that ``creme``
contains.

"""
import typing

from .anomaly import AnomalyDetector
from .classifier import Classifier
from .classifier import BinaryClassifier
from .classifier import MultiClassifier
from .clusterer import Clusterer
from .ensemble import Ensemble
from .estimator import Estimator
from .multi_output import MultiOutputClassifier
from .multi_output import MultiOutputRegressor
from .regressor import Regressor
from .transformer import Transformer
from .wrapper import Wrapper


__all__ = [
    'AnomalyDetector',
    'BinaryClassifier',
    'Classifier',
    'Clusterer',
    'Ensemble',
    'Estimator',
    'MultiClassifier',
    'MultiOutputClassifier',
    'MultiOutputRegressor',
    'Regressor',
    'Transformer',
    'Wrapper'
]
