import math

from .. import utils

from . import base


class Max(base.Univariate):
    """Running max.

    Attributes:
        max (float): The running max.

    """

    def __init__(self):
        self.max = -math.inf

    @property
    def name(self):
        return 'max'

    def update(self, x):
        if x > self.max:
            self.max = x
        return self

    def get(self):
        return self.max


class RollingMax(base.Univariate, utils.SortedWindow):
    """Running max over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> rolling_max = stats.RollingMax(window_size=2)
            >>> for x in X:
            ...     print(rolling_max.update(x).get())
            1
            1
            3
            3
            2
            2

    """

    def __init__(self, window_size):
        super().__init__(size=window_size)

    @property
    def name(self):
        return f'rolling_{self.window_size}_max'

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        return self[-1]
