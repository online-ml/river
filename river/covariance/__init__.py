"""Online estimation of covariance and precision matrices.

A covariance matrix summarises how a set of variables move together. It is the engine behind
portfolio risk, anomaly detection (via the Mahalanobis distance), Gaussian models, and many
dimensionality-reduction methods. This module estimates it (and its inverse, the precision
matrix) incrementally from a stream, without storing the data. See each estimator's docstring for
what it does and when to reach for it.

The estimators are dict-native: `update(x)` takes a mapping and the `matrix` is a dict of pairwise
values. Most also expose an `update_many` method for mini-batches of any narwhals-compatible
dataframe.

"""

from __future__ import annotations

from .emp import EmpiricalCovariance, EmpiricalPrecision
from .ewa import (
    EwaCovariance,
    EwaPrecision,
    LedoitWolfCovariance,
    OASCovariance,
    ShrunkCovariance,
)

__all__ = [
    "EmpiricalCovariance",
    "EmpiricalPrecision",
    "EwaCovariance",
    "EwaPrecision",
    "LedoitWolfCovariance",
    "OASCovariance",
    "ShrunkCovariance",
]
