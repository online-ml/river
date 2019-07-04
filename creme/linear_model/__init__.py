"""
Generalized linear models optimized with online gradient descent from :mod:`creme.optim`.
"""
from .fm import FMRegressor
from .glm import LinearRegression
from .glm import LogisticRegression
from .pa import PAClassifier
from .pa import PARegressor
from .softmax import SoftmaxRegression


__all__ = [
    'FMRegressor',
    'GLMRegressor',
    'LinearRegression',
    'LogisticRegression',
    'PAClassifier',
    'PARegressor',
    'SoftmaxRegression'
]
