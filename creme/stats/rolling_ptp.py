from . import base
from . import rolling_max
from . import rolling_min


class RollingPeakToPeak(base.Univariate):
    """Running peak to peak (max - min) over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Attributes:
        max (stars.RollingMax): The running rolling max.
        min (stars.RollingMin): The running rolling min.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> ptp = stats.RollingPeakToPeak(window_size=2)
            >>> for x in X:
            ...     print(ptp.update(x).get())
            0
            5
            7
            5
            4
            1

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.max = rolling_max.RollingMax(window_size)
        self.min = rolling_min.RollingMin(window_size)

    @property
    def name(self):
        return f'rolling_{self.window_size}_ptp'

    def update(self, x):
        self.max.update(x)
        self.min.update(x)
        return self

    def get(self):
        return self.max.get() - self.min.get()
