"""
Generalized linear models optimized through stochastic gradient descent using
the :mod:`creme.optim` module.
"""
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression


__all__ = ['LinearRegression', 'LogisticRegression']
