from . import base
from . import _window


class RollingQuantile(base.RunningStatistic, _window.SortedWindow):
    """Running quantile over a window.

    Parameters:
        window_size (int): Size of the window.
        quantile (float): Desired quantile, must be between 0 and 1.

    Example:

    ::

        >>> from creme import stats
        >>> import numpy as np

        >>> rolling_quantile = stats.RollingQuantile(window_size = 100,
        ...                                          quantile = 0.5)

        >>> for i in range(0,1001):
        ...     _ = rolling_quantile.update(i)
        ...     if i%100 == 0:
        ...         print(rolling_quantile.get())
        0
        50
        150
        250
        350
        450
        550
        650
        750
        850
        950

    References:

    1. `Left sorted <https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>`_

    """

    def __init__(self, window_size, quantile=0.5):
        super().__init__(window_size)
        self.quantile = quantile
        self.idx = int(round(self.quantile * self.window_size + 0.5)) - 1

    @property
    def name(self):
        return f'rolling_{self.window_size}_quantile'

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        if len(self) < self.window_size:
            idx = int(round(self.quantile * len(self) + 0.5)) - 1
            return self.sorted_window[idx]
        return self.sorted_window[self.idx]
