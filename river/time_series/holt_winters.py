from __future__ import annotations

import operator
import statistics
from collections import deque

from river import time_series

__all__ = ["HoltWinters"]


class Component(deque):
    ...


class AdditiveLevel(Component):
    def __init__(self, alpha):
        super().__init__([], maxlen=2)
        self.alpha = alpha

    def update(self, y, trend, season):
        self.append(
            self.alpha * (y - (season[-season.seasonality] if season else 0))
            + (1 - self.alpha) * (self[-1] + (trend[-1] if trend else 0))
        )


class MultiplicativeLevel(Component):
    def __init__(self, alpha):
        super().__init__([], maxlen=2)
        self.alpha = alpha

    def update(self, y, trend, season):
        self.append(
            self.alpha * (y / (season[-season.seasonality] if season else 1))
            + (1 - self.alpha) * (self[-1] + (trend[-1] if trend else 0))
        )


class Trend(Component):
    def __init__(self, beta):
        super().__init__([], maxlen=2)
        self.beta = beta

    def update(self, y, level):
        self.append(self.beta * (level[-1] - level[-2]) + (1 - self.beta) * self[-1])


class AdditiveSeason(Component):
    def __init__(self, gamma, seasonality):
        super().__init__([], maxlen=seasonality + 1)
        self.gamma = gamma
        self.seasonality = seasonality

    def update(self, y, level, trend):
        self.append(
            self.gamma * (y - level[-2] - trend[-2]) + (1 - self.gamma) * self[-self.seasonality]
        )


class MultiplicativeSeason(Component):
    def __init__(self, gamma, seasonality):
        super().__init__([], maxlen=seasonality + 1)
        self.gamma = gamma
        self.seasonality = seasonality

    def update(self, y, level, trend):
        self.append(
            self.gamma * y / (level[-2] + trend[-2]) + (1 - self.gamma) * self[-self.seasonality]
        )


class HoltWinters(time_series.base.Forecaster):
    r"""Holt-Winters forecaster.

    This is a standard implementation of the Holt-Winters forecasting method. Certain
    parametrisations result in special cases, such as simple exponential smoothing.

    Optimal parameters and initialisation values can be determined in a batch setting. However, in
    an online setting, it is necessary to wait and observe enough values. The first
    `k = max(2, seasonality)` values are indeed used to initialize the components.

    **Level initialization**

    $$l = \frac{1}{k} \sum_{i=1}{k} y_i$$

    **Trend initialization**

    $$t = \frac{1}{k - 1} \sum_{i=2}{k} y_i - y_{i-1}$$

    **Trend initialization**

    $$s_i = \frac{y_i}{k}$$

    Parameters
    ----------
    alpha
        Smoothing parameter for the level.
    beta
        Smoothing parameter for the trend.
    gamma
        Smoothing parameter for the seasonality.
    seasonality
        The number of periods in a season. For instance, this should be 4 for quarterly data,
        and 12 for yearly data.
    multiplicative
        Whether or not to use a multiplicative formulation.

    Examples
    --------

    >>> from river import datasets
    >>> from river import metrics
    >>> from river import time_series

    >>> dataset = datasets.AirlinePassengers()

    >>> model = time_series.HoltWinters(
    ...     alpha=0.3,
    ...     beta=0.1,
    ...     gamma=0.6,
    ...     seasonality=12,
    ...     multiplicative=True
    ... )

    >>> metric = metrics.MAE()

    >>> time_series.evaluate(
    ...     dataset,
    ...     model,
    ...     metric,
    ...     horizon=12
    ... )
    +1  MAE: 25.899087
    +2  MAE: 26.26131
    +3  MAE: 25.735903
    +4  MAE: 25.625678
    +5  MAE: 26.093842
    +6  MAE: 26.90249
    +7  MAE: 28.634398
    +8  MAE: 29.284769
    +9  MAE: 31.018351
    +10 MAE: 32.252349
    +11 MAE: 33.518946
    +12 MAE: 33.975057

    References
    ----------
    [^1]: [Exponential smoothing — Wikipedia](https://www.wikiwand.com/en/Exponential_smoothing)
    [^2]: [Exponential smoothing — Forecasting: Principles and Practice](https://otexts.com/fpp2/expsmooth.html)
    [^3]: [What is Exponential Smoothing? — Engineering statistics handbook](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc43.htm)

    """

    def __init__(
        self,
        alpha,
        beta=None,
        gamma=None,
        seasonality=0,
        multiplicative=False,
    ):
        if seasonality and gamma is None:
            raise ValueError("gamma must be set if seasonality is set")

        if gamma and beta is None:
            raise ValueError("beta must be set if gamma is set")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonality = seasonality
        self.multiplicative = multiplicative
        self.level = MultiplicativeLevel(alpha) if multiplicative else AdditiveLevel(alpha)
        self.trend = Trend(beta) if beta else None
        self.season = (
            (
                MultiplicativeSeason(gamma, seasonality)
                if multiplicative
                else AdditiveSeason(gamma, seasonality)
            )
            if seasonality
            else None
        )
        self._first_values = []
        self._initialized = False

    def learn_one(self, y, x=None):
        if self._initialized:
            self.level.update(y, self.trend, self.season)
            if self.trend is not None:
                self.trend.update(y, self.level)
            if self.season is not None:
                self.season.update(y, self.level, self.trend)
            return self

        self._first_values.append(y)
        if len(self._first_values) < max(2, self.seasonality):
            return self

        # The components can be initialized now that enough values have been observed
        self.level.append(statistics.mean(self._first_values))
        diffs = [b - a for a, b in zip(self._first_values[:-1], self._first_values[1:])]
        if self.trend is not None:
            self.trend.append(statistics.mean(diffs))
        if self.season is not None:
            self.season.extend([y / self.level[-1] for y in self._first_values])

        self._initialized = True

        return self

    def forecast(self, horizon, xs=None):
        op = operator.mul if self.multiplicative else operator.add
        return [
            op(
                self.level[-1] + ((h + 1) * self.trend[-1] if self.trend else 0),
                (
                    self.season[-self.seasonality + h % self.seasonality]
                    if self.season
                    else (1 if self.multiplicative else 0)
                ),
            )
            for h in range(horizon)
        ]
