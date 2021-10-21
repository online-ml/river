from collections import deque
from .base import Forecaster


class Component(deque):
    ...


class Level(Component):

    def __init__(self, alpha, damping):
        super().__init__([32.2597], maxlen=2)
        self.alpha = alpha
        self.damping = damping

    def update(self, y, trend, seasonality):
        self.append(
            self.alpha * (y - seasonality[-seasonality.m]) +
            (1 - self.alpha) * (self[-1] + self.damping * trend[-1])
        )


class Trend(Component):

    def __init__(self, beta, damping):
        super().__init__([0.7014], maxlen=2)
        self.beta = beta
        self.damping = damping

    def update(self, y, level):
        self.append(self.beta * (level[-1] - level[-2]) + (1 - self.beta) * self.damping * self[-1])


class Seasonality(Component):

    def __init__(self, gamma, m):
        super().__init__([9.6962, -9.3132, -1.6935, 1.3106], maxlen=m + 1)
        self.gamma = gamma
        self.m = m

    def update(self, y, level, trend):
        self.append(self.gamma * (y - level[-1] - trend[-1]) + (1 - self.gamma) * self[-self.m])


class HoltWinters(Forecaster):
    def __init__(self, alpha, beta, gamma, m, damping=1):
        self.level = Level(alpha, damping)
        self.trend = Trend(beta, damping)
        self.seasonality = Seasonality(gamma, m)
        self.damping = damping
        self.m = m

    def learn_one(self, y):
        self.seasonality.update(y, self.level, self.trend)
        self.level.update(y, self.trend, self.seasonality)
        self.trend.update(y, self.level)
        return self

    def forecast(self, horizon):
        return [
            self.level[-1] + (h + 1) * self.trend[-1] + self.seasonality[-self.m + 1]
            for h in range(horizon)
        ]
