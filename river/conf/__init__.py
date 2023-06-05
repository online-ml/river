"""Conformal predictions. This modules contains wrappers to enable conformal predictions on any
regressor or classifier."""
from __future__ import annotations

from .interval import Interval
from .jackknife import RegressionJackknife

__all__ = [
    "Interval",
    "RegressionJackknife",
]
