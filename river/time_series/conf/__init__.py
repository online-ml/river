"""Conformal predictions. This modules contains wrappers to enable conformal predictions on any
regressor or classifier."""
from .base import Interval
from .jackknife import RegressionJackknife
from .gaussian import Gaussian

__all__ = [
    "base",
    "Interval",
    "Gaussian",
    "RegressionJackknife",
]
