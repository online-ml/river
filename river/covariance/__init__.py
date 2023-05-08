"""Online estimation of covariance and precision matrices."""
from __future__ import annotations

from .emp import EmpiricalCovariance, EmpiricalPrecision

__all__ = ["EmpiricalCovariance", "EmpiricalPrecision"]
