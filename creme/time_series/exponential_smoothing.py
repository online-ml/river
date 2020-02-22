from typing import List, Optional
import collections  # replace by creme utils windows
from . import base


def _discount_sum(phi: float, h: int) -> list:
    return sum([phi**i for i in range(1, h + 1)])


class SimpleExponentialSmoothing(base.Forecaster):
    """
     Simple exponential smoothing.
     Mathematically, it is defined as:
     Forecast equation
     .. math:: \hat{y}_{t+h | t}=\ell_{t}
     Smoothing equation
     .. math:: \ell_{t}=\alpha y_{t}+(1-\alpha) \ell_{t-1} 

    Parameters:
        alpha (float): The smoothing parameter for the level, 0 ≤ alpha ≤ 1 . Defaults to `0.5`.
        l0 (float): Initialization value for the level. Defaults to `0`.

    Example:
        ::
            >>> #TODO

    References:
        1. `Simple exponential smoothing <https://otexts.com/fpp2/ses.html>`_
    """

    def __init__(self, alpha=0.5, l0: float = 0):

        if 0 <= alpha <= 1:
            self.alpha = alpha
        else:
            raise ValueError(f'The value of alpha must be between [0, 1]')

        self.lt = l0

    def fit_one(self, y: float):
        self.st = self.alpha * y + (1 - self.alpha) * self.lt

        return self

    def forecast(self, horizon: int) -> list:
        return [self.lt for _ in range(horizon)]


class HoltLinearTrend(base.Forecaster):
    """
    #TODO : Docstring + test

    Parameters:
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.

    References:
        1. `Holt’s linear trend method <https://otexts.com/fpp2/holt.html>`_
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 l0: float = 0,
                 b0: float = 0):

        if (0 <= alpha <= 1) and (0 <= beta <= 1):
            self.alpha = alpha
            self.beta = beta
        else:
            raise ValueError(
                'The value of alpha and beta must be between [0, 1]')

        self.lt = l0
        self.bt = b0

    def fit_one(self, y: float):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * y + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1

        return self

    def forecast(self, horizon: int) -> float:
        return [self.lt + (h * self.bt) for h in range(horizon)]


class DampedTrend(base.Forecaster):
    """
    #TODO : Docstring + test

    Parameters:
        alpha (float): ... .Defaults to `0.5`.
        beta (float): ... .Defaults to `0.5`.
        phi (float): ... .Defaults to `0.5`.

    References:
        1. `Damped trend methods <https://otexts.com/fpp2/holt.html>`_
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 phi: float = 0.5,
                 l0: float = 0,
                 b0: float = 0):

        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= phi <= 1):
            self.alpha = alpha
            self.beta = beta
            self.phi = phi
        else:
            raise ValueError(
                'The value of alpha and beta must be between [0, 1] and phi must be between (0, 1)'
            )

        self.lt = l0
        self.bt = b0

    def fit_one(self, y):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * y + (1 - self.alpha) * (lt_1 + self.phi * bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (
            1 - self.beta) * bt_1 * self.phi

        return self

    def forecast(self, horizon: int) -> list:
        return [
            self.lt + (_discount_sum(self.phi, h) * self.bt)
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
                 gamma: float = 0.5,
                 s: Optional[List[float]] = None,  # We can discuss for this choice 
                 l0: float = 0,
                 b0: float = 0):

        self.m = m

        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        else:
            raise ValueError(
                'The value of alpha, beta and gamma must be between [0, 1]')

        # init s : [s_{t-m}, s_{t-m+1}, ..., s_{t}]
        if s is None:
            self.s = collections.deque(maxlen=m)
            for i in range(self.m):
                self.s.append(0)
        else:
            if len(s) == m:
                self.s = collections.deque(s, maxlen=m)
            else:
                raise ValueError('s must have a size of m obligatory')

        self.lt = l0
        self.bt = b0

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
                 gamma: float = 0.5,
                 s: Optional[List[float]] = None,  # We can discuss for this choice 
                 l0: float = 0.5,
                 b0: float = 0.5):

        self.m = m

        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
        else:
            raise ValueError(
                'The value of alpha, beta and gamma must be between [0, 1]')

        # init s : [s_{t-m}, s_{t-m+1}, ..., s_{t}]
        if s is None:
            self.s = collections.deque(maxlen=m)
            for i in range(self.m):
                self.s.append(1)
        else:
            if len(s) == m:
                self.s = collections.deque(s, maxlen=m)
            else:
                raise ValueError('s must have a size of m obligatory')

        self.lt = l0
        self.bt = b0

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


class HoltWinterDamped(base.Forecaster):
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
                 phi: float = 0.5,
                 s: Optional[List[float]] = None,  # We can discuss for this choice 
                 l0: float = 0.5,
                 b0: float = 0.5):

        self.m = m
        if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <=
                                                       1) and (0 <= phi <= 1):
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.phi = phi
        else:
            raise ValueError(
                'The value of alpha, beta and gamma must be between [0, 1] and phi must be between (0, 1)'
            )

        # init s : [s_{t-m}, s_{t-m+1}, ..., s_{t}]
        if s is None:
            self.s = collections.deque(maxlen=m)
            for i in range(self.m):
                self.s.append(1)
        else:
            if len(s) == m:
                self.s = collections.deque(s, maxlen=m)
            else:
                raise ValueError('s must have a size of m obligatory')

        self.lt = l0
        self.bt = b0

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
        return (self.lt + _discount_sum(self.phi, h) * self.bt
                ) * self.s[h - self.m * (((h - 1) // self.m) + 1)]
