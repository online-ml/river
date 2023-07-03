from __future__ import annotations

import math

from river import base, compose, stats

__all__ = [
    "TargetMinMaxScaler",
    "TargetStandardScaler",
]


def safe_div(a, b):
    """Returns a if b is nil, else divides a by b.

    When scaling, sometimes a denominator might be nil. For instance, during standard scaling
    the denominator can be nil if a feature has no variance.

    """
    return a / b if b else 0.0


class TargetStandardScaler(compose.TargetTransformRegressor):
    """Applies standard scaling to the target.

    Parameters
    ----------
    regressor
        Regression model to wrap.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     preprocessing.TargetStandardScaler(
    ...         regressor=linear_model.LinearRegression(intercept_lr=0.15)
    ...     )
    ... )
    >>> metric = metrics.MSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 2.005999

    """

    def __init__(self, regressor: base.Regressor):
        self.var = stats.Var()
        super().__init__(regressor=regressor, func=self._scale, inverse_func=self._unscale)

    def _update(self, y):
        self.var.update(y)

    def _scale(self, y):
        return safe_div(y - self.var.mean.get(), self.var.get() ** 0.5)

    def _unscale(self, y):
        return y * self.var.get() ** 0.5 + self.var.mean.get()


class TargetMinMaxScaler(compose.TargetTransformRegressor):
    """Applies min-max scaling to the target.

    Parameters
    ----------
    regressor
        Regression model to wrap.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     preprocessing.TargetMinMaxScaler(
    ...         regressor=linear_model.LinearRegression(intercept_lr=0.15)
    ...     )
    ... )
    >>> metric = metrics.MSE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MSE: 2.018905

    """

    def __init__(self, regressor: base.Regressor):
        self.min = stats.Min()
        self.max = stats.Max()
        super().__init__(regressor=regressor, func=self._scale, inverse_func=self._unscale)

    def _update(self, y):
        self.min.update(y)
        self.max.update(y)

    def _scale(self, y):
        return safe_div(y - self.min.get(), self.max.get() - self.min.get())

    def _unscale(self, y):
        if self.min.get() == math.inf:
            return y
        return y * (self.max.get() - self.min.get()) + self.min.get()
