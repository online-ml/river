"""
Generalized linear models optimized with online gradient descent using the
:mod:`creme.optim` module.
"""
from .fm import FMRegressor
from .lin_reg import LinearRegression
from .log_reg import LogisticRegression


__all__ = [
    'FMRegressor',
    'LinearRegression',
    'LogisticRegression'
]
