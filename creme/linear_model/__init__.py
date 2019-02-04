"""
Generalized linear models optimized through stochastic gradient descent using
the :mod:`creme.optim` module.
"""
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .pa import PassiveAggressiveClassifier
from .pa import PassiveAggressiveRegressor
from .pa import PassiveAggressiveIClassifier
from .pa import PassiveAggressiveIRegressor
from .pa import PassiveAggressiveIIClassifier
from .pa import PassiveAggressiveIIRegressor


__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'PassiveAggressiveClassifier',
    'PassiveAggressiveRegressor',
    'PassiveAggressiveIClassifier',
    'PassiveAggressiveIRegressor',
    'PassiveAggressiveIIClassifier',
    'PassiveAggressiveIIRegressor'
]
