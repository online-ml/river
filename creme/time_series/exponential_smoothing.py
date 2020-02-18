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
