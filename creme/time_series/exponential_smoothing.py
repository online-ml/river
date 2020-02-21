# INIT LO BO ETC ......
from typing import Optional
import collections  # replace by creme utils windows
from . import base


class _DampedSmoothing:
    @staticmethod
    def _discount_sum(phi: float, h: int) -> list:
        return sum([phi**i for i in range(1, h + 1)])


class ExponentialSmoothing(base.Forecaster):
    """
     #TODO : Docstring + test

    Parameters:
        alpha (float): Defaults to `0.5`.

    References:
        1. `Simple exponential smoothing <https://otexts.com/fpp2/ses.html>`_
    """

    def __init__(self, alpha=0.5, s0: Optional[float] = None):

        if 0 < self.alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError(f'The value of alpha must be between (0, 1]')

        if s0 is None:
            self.st = 0
        else:
            self.st = s0

    def fit_one(self, y: float):
        self.st = self.alpha * y + (1 - self.alpha) * self.st

        return self

    def forecast(self, horizon: int) -> list:
        return [self.st for _ in range(horizon)]


class HoltLinearTrend(base.Forecaster):
    """
    #TODO : Docstring + test

    Parameters:
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.

    References:
        1. `Holt’s linear trend method <https://otexts.com/fpp2/holt.html>`_
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.lt = 0
        self.bt = 0

    def fit_one(self, y):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * y + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1

        return self

    def forecast(self, horizon: int) -> float:
        return [self.lt + (h * self.bt) for h in range(horizon)]


class DampedTrend(base.Forecaster, _DampedSmoothing):
    """
    #TODO : Docstring + test

    Parameters:
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.
        phi (float): ... .Defaults to `0.5`.

    References:
        1. `Damped trend methods <https://otexts.com/fpp2/holt.html>`_
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 phi: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.lt = 0
        self.bt = 0

    def fit_one(self, y):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * y + (1 - self.alpha) * (lt_1 + self.phi * bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (
            1 - self.beta) * bt_1 * self.phi

        return self

    def forecast(self, horizon: int) -> list:
        return [
            self.lt + (self._discount_sum(self.phi, h) * self.bt)
            for h in range(horizon)
        ]


class HoltWinterAdditive(base.Forecaster):
    """
    #TODO : Docstring + test

    Parameters:
        m (int): ...
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.
        gamma (float): ... .Defaults to `0.5`.

    References:
        1. `Holt-Winters’ additive method <https://otexts.com/fpp2/holt-winters.html>`_
    """

    def __init__(self,
                 m: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.5):

        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # init s : [s_{t-m}, s_{t-m+1}, ..., s_{t}]
        self.s = collections.deque(maxlen=m)
        for i in range(self.m):
            self.s.append(0)

        self.lt = 0
        self.bt = 0

    def fit_one(self, y: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * (y - st_m) + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1
        st = self.gamma * (y - lt_1 - bt_1) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def forecast(self, horizon: int) -> list:
        return [
            self.lt + h * self.bt + self.s[h - self.m * ((
                (h - 1) // self.m) + 1)] for h in range(horizon)
        ]


class HoltWinterMultiplicative(base.Forecaster):
    """
    #TODO : Docstring + test

    Parameters:
        m (int): ...
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.
        gamma (float): ... .Defaults to `0.5`.

    References:
        1. `Holt-Winters’ multiplicative method <https://otexts.com/fpp2/holt-winters.html>`_
    """

    def __init__(self,
                 m: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.5):

        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # init st
        self.s = collections.deque(maxlen=m)
        for i in range(self.m):
            self.s.append(1)

        self.lt = 0.5
        self.bt = 0.5

    def fit_one(self, y: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        # replace by safe division
        self.lt = self.alpha * (y / st_m) + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1
        st = self.gamma * (y / (lt_1 + bt_1)) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def forecast(self, horizon: int) -> list:
        return [(self.lt + h * self.bt) * self.s[h - self.m * ((
            (h - 1) // self.m) + 1)] for h in range(horizon)]


class HoltWinterDamped(base.Forecaster, _DampedSmoothing):
    """
        #TODO : Docstring + test

    Parameters:
        m (int): ...
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.
        gamma (float): ... .Defaults to `0.5`.
        phi (float): ... .Defaults to `0.5`.

    References:
        1. `Holt-Winters’ damped method <https://otexts.com/fpp2/holt-winters.html>`_
    """

    def __init__(self,
                 m: int,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 gamma: float = 0.5,
                 phi: float = 0.5):

        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phi = phi

        # init st
        self.s = collections.deque(maxlen=m)
        for i in range(self.m):
            self.s.append(1)

        self.lt = 0.5
        self.bt = 0.5

    def fit_one(self, y: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        # replace by safe division
        self.lt = self.alpha * (y / st_m) + (1 - self.alpha) * (
            lt_1 + self.phi * bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (
            1 - self.beta) * self.phi * bt_1
        st = self.gamma * (y /
                           (lt_1 + self.phi * bt_1)) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def forecast(self, h: int) -> float:
        return (self.lt + self._discount_sum(self.phi, h) * self.bt
                ) * self.s[h - self.m * (((h - 1) // self.m) + 1)]
