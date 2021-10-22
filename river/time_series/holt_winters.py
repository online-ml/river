import operator
import statistics
from collections import deque

from .base import Forecaster

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
            self.gamma * (y - level[-2] - trend[-2])
            + (1 - self.gamma) * self[-self.seasonality]
        )


class MultiplicativeSeason(Component):
    def __init__(self, gamma, seasonality):
        super().__init__([], maxlen=seasonality + 1)
        self.gamma = gamma
        self.seasonality = seasonality

    def update(self, y, level, trend):
        self.append(
            self.gamma * y / (level[-2] + trend[-2])
            + (1 - self.gamma) * self[-self.seasonality]
        )


class HoltWinters(Forecaster):
    """

    References
    ----------
    [^1] [Wikipedia page on exponential smoothing](https://www.wikiwand.com/en/Exponential_smoothing)

    """

    def __init__(
        self,
        alpha,
        beta=None,
        gamma=None,
        seasonality: int = None,
        multiplicative=False,
    ):
        self.level = (
            MultiplicativeLevel(alpha) if multiplicative else AdditiveLevel(alpha)
        )
        self.trend = Trend(beta) if beta else None
        self.season = (
            (
                MultiplicativeSeason(gamma, seasonality)
                if multiplicative
                else AdditiveSeason(gamma, seasonality)
            )
            if (gamma or seasonality)
            else None
        )
        self.seasonality = seasonality
        self.multiplicative = multiplicative
        self._first_values = []
        self._initialized = False

    def learn_one(self, y, x=None):
        if self._initialized:
            self.level.update(y, self.trend, self.season)
            if self.trend:
                self.trend.update(y, self.level)
            if self.season:
                self.season.update(y, self.level, self.trend)
            return self

        self._first_values.append(y)
        if len(self._first_values) < max(2, self.seasonality):
            return self

        # The components can be initialized now that enough values have been observed
        self.level.append(statistics.mean(self._first_values))
        self.trend.append(
            statistics.mean(
                [b - a for a, b in zip(self._first_values[:-1], self._first_values[1:])]
            )
        )

        self._initialized = True

        return self

    def forecast(self, horizon, xs=None):
        op = operator.mul if self.multiplicative else operator.add
        return [
            op(
                self.level[-1] + ((h + 1) * self.trend[-1] if self.trend else 0),
                (
                    self.season[-self.seasonality + h % self.seasonality]
                    if self.seasonality
                    else 0
                ),
            )
            for h in range(horizon)
        ]
