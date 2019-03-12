"""
Generalized linear models optimized through stochastic gradient descent using
the :mod:`creme.optim` module.
"""
from .fm import FMRegressor
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression


__all__ = [
    'FMRegressor',
    'LinearRegression',
    'LogisticRegression'
]
