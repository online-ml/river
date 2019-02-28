from . import rolling_window
from . import rolling_mean


class RollingVariance:
    """Compute the windowed rolling variance.

    Attributes:
        window_size (int): Size of the rolling window.
        ddof (int): Factor to correct variance.
        sum_square (float): Sum of the square of the value used to update RollingVariance.
        rolling_mean (RollingMean): Compute rolling windowed mean.
        rolling_window (RollingWindow): Store K current values.

    >>> import creme
    >>> import pandas as pd

    >>> X = pd.Series([1, 4, 2, -4, -8, 0])

    >>> print(X.rolling(2).var())
    0     NaN
    1     4.5
    2     2.0
    3    18.0
    4     8.0
    5    32.0
    dtype: float64

    >>> rolling_variance = creme.stats.RollingVariance(ddof=1, window_size=2)
    >>> for x in X:
    ...     print(rolling_variance.update(x).get())
    0.0
    4.5
    2.0
    18.0
    8.0
    32.0


    >>> print(X.rolling(3).var())
    0          NaN
    1          NaN
    2     2.333333
    3    17.333333
    4    25.333333
    5    16.000000
    dtype: float64

    >>> rolling_variance = creme.stats.RollingVariance(ddof=1, window_size=3)
    >>> for x in X:
    ...     print(rolling_variance.update(x).get())
    0.0
    4.5
    2.333333...
    17.33333...
    25.33333...
    16.0

    """

    def __init__(self, window_size, ddof=1):
        self.sum_square = 0
        self.ddof = ddof
        self.window_size = window_size
        self.rolling_mean = rolling_mean.RollingMean(window_size)
        self.rolling_window = rolling_window.RollingWindow(window_size)

    @property
    def name(self):
        return 'rolling_variance'

    def update(self, x):
        if len(self.rolling_window.get()) + 1 > self.window_size:
            self.sum_square -= self.rolling_window.get()[0] ** 2

        self.sum_square += x**2
        self.rolling_mean.update(x)
        self.rolling_window.update(x)
        return self

    def _get_correction_factor(self):
        if self.rolling_mean.count.get() > self.ddof:
            return self.rolling_mean.count.get() / (self.rolling_mean.count.get() - self.ddof)
        else:
            return 1

    def get(self):
        correction_factor = self._get_correction_factor()
        if self.sum_square > 0:
            variance = ((self.sum_square / self.rolling_mean.count.get()
                         ) - self.rolling_mean.get() ** 2)
        else:
            variance = 0
        return correction_factor * variance
