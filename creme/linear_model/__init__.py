"""
Generalized linear models optimized with online gradient descent using the
:mod:`creme.optim` module.
"""
from .fm import FMRegressor
from .lin_reg import LinearRegression
from .log_reg import LogisticRegression
from .pa import PAClassifier
from .pa import PARegressor
from .softmax import SoftmaxRegression


__all__ = [
    'FMRegressor',
    'LinearRegression',
    'LogisticRegression',
    'PAClassifier',
    'PARegressor',
    'SoftmaxRegression'
]
