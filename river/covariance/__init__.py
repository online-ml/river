"""Online estimation of covariance and precision matrices."""

from .emp import EmpiricalCovariance, EmpiricalPrecision

__all__ = ["EmpiricalCovariance", "EmpiricalPrecision"]
