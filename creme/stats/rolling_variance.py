from . import base
from . import rolling_mean


class RollingVariance(base.RunningStatistic):
    """Running variance over a window.

    Parameters:
        window_size (int): Size of the rolling window.
        ddof (int): Factor used to correct the variance.

    Attributes:
        sos (float): Sum of squares.
        rolling_mean (RollingMean)

    >>> import creme

    >>> X = [1, 4, 2, -4, -8, 0]

    >>> rolling_variance = creme.stats.RollingVariance(ddof=1, window_size=2)
    >>> for x in X:
    ...     print(rolling_variance.update(x).get())
    0.0
    4.5
    2.0
    18.0
    8.0
    32.0

    >>> rolling_variance = creme.stats.RollingVariance(ddof=1, window_size=3)
    >>> for x in X:
    ...     print(rolling_variance.update(x).get())
    0.0
    4.5
    2.333333...
    17.333333...
    25.333333...
    16.0

    """

    def __init__(self, window_size, ddof=1):
        self.window_size = window_size
        self.ddof = ddof
        self.sos = 0
        self.rolling_mean = rolling_mean.RollingMean(window_size)

    @property
    def name(self):
        return f'rolling_{self.window_size}_variance'

    def update(self, x):
        if len(self.rolling_mean) >= self.window_size:
            self.sos -= self.rolling_mean.window[0] ** 2

        self.sos += x * x
        self.rolling_mean.update(x)
        return self

    @property
    def correction_factor(self):
        n = len(self.rolling_mean)
        if n > self.ddof:
            return n / (n - self.ddof)
        return 1

    def get(self):
        variance = (self.sos / len(self.rolling_mean)) - self.rolling_mean.get() ** 2
        return self.correction_factor * variance
