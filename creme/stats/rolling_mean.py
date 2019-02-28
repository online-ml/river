from . import mean
from . import rolling_window


class RollingMean(mean.Mean):
    """Calculate the rolling average with a given window size.

    Attributes:
        window_size (int): Size of the rolling window.
        current_window (collections.deque): Store values that are in the current window.

    >>> import creme
    >>> import pandas as pd

    >>> X = pd.Series([1, 2, 3, 4, 5, 6])

    # Pandas:
    >>> print(X.rolling(2).mean())
    0    NaN
    1    1.5
    2    2.5
    3    3.5
    4    4.5
    5    5.5
    dtype: float64

    >>> rolling_mean = creme.stats.RollingMean(window_size=2)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5

    # Pandas:
    >>> print(X.rolling(3).mean())
    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    5    5.0
    dtype: float64

    >>> rolling_mean = creme.stats.RollingMean(window_size=3)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.0
    3.0
    4.0
    5.0

    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.rolling_window = rolling_window.RollingWindow(window_size)

    @property
    def name(self):
        return 'rolling_mean'

    def update(self, x):
        if len(self.rolling_window.get()) + 1 > self.window_size:
            self.mean = (self.mean * self.count.get() -
                         self.rolling_window.get()[0]) / (self.count.get() - 1)
            self.count.desincrement()

        self.rolling_window.update(x)
        super().update(x)
        return self

    def get(self):
        return self.mean
