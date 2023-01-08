"""Conformal predictions. This modules contains wrappers to enable conformal predictions on any
regressor or classifier."""
from .base import Interval, RegressionInterval
from .jackknife import RegressionJackknife

__all__ = [
    "base",
    "Interval",
    "RegressionInterval",
    "RegressionJackknife",
]
