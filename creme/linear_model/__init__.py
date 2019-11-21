"""Linear models."""
from .fm import FMClassifier
from .fm import FMRegressor
from .glm import LinearRegression
from .glm import LogisticRegression
from .pa import PAClassifier
from .pa import PARegressor
from .softmax import SoftmaxRegression


__all__ = [
    'FMClassifier',
    'FMRegressor',
    'LinearRegression',
    'LogisticRegression',
    'PAClassifier',
    'PARegressor',
    'SoftmaxRegression'
]
