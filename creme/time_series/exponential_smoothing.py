import collections  # replace by creme utils windows


class ExponentialSmoothing:
    """
     #TODO : Docstring + test

    Parameters:
        alpha (float): Defaults to `0.5`.

    References:
        1. `Simple exponential smoothing <https://otexts.com/fpp2/ses.html>`_
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.st = 0

    def update(self, x: float):
        self.st = self.alpha * x + (1 - self.alpha) * self.st

        return self

    def get(self) -> float:
        return self.st


class HoltES:
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

    def update(self, x):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * x + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1

        return self

    def get(self, h: int) -> float:
        return self.lt + (h * self.bt)


class DampedES:
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

    def update(self, x):

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * x + (1 - self.alpha) * (lt_1 + self.phi * bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (
            1 - self.beta) * bt_1 * self.phi

        return self

    def get(self, h: int) -> float:
        discount_sum = self._compute_discount_sum(self.phi, h)
        return self.lt + (discount_sum * self.bt)

    @staticmethod
    def _compute_discount_sum(phi: float, h: int) -> float:
        return sum([phi**i for i in range(1, h + 1)])


class HoltWinterAdditive:
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

    def update(self, x: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        self.lt = self.alpha * (x - st_m) + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1
        st = self.gamma * (x - lt_1 - bt_1) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def get(self, h: int) -> float:
        k = (h - 1) // self.m
        return self.lt + h * self.bt + self.s[h - self.m * (k + 1)]


class HoltWinterMultiplicative:
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

    def update(self, x: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        # replace by safe division
        self.lt = self.alpha * (x / st_m) + (1 - self.alpha) * (lt_1 + bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (1 - self.beta) * bt_1
        st = self.gamma * (x / (lt_1 + bt_1)) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def get(self, h: int) -> float:
        k = (h - 1) // self.m
        return (self.lt + h * self.bt) * self.s[h - self.m * (k + 1)]


class HoltWinterDamped:
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

    def update(self, x: float):

        st_1 = self.s[-1]
        st_m = self.s[-self.m]

        lt_1 = self.lt
        bt_1 = self.bt

        # replace by safe division
        self.lt = self.alpha * (x / st_m) + (1 - self.alpha) * (
            lt_1 + self.phi * bt_1)
        self.bt = self.beta * (self.lt - lt_1) + (
            1 - self.beta) * self.phi * bt_1
        st = self.gamma * (x /
                           (lt_1 + self.phi * bt_1)) + (1 - self.gamma) * st_m
        self.s.append(st)

        return self

    def get(self, h: int) -> float:
        k = (h - 1) // self.m
        discount_sum = self._compute_discount_sum(self.phi, h)
        return (self.lt + discount_sum * self.bt) * self.s[h - self.m *
                                                           (k + 1)]

    @staticmethod
    def _compute_discount_sum(phi: float, h: int) -> float:
        return sum([phi**i for i in range(1, h + 1)])
