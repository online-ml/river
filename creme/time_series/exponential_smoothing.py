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

