import math

from .. import utils

from . import base


class Min(base.Univariate):
    """Running min.

    Attributes:
        min (float): The running min.

    """

    def __init__(self):
        self.min = math.inf

    def update(self, x):
        if x < self.min:
            self.min = x
        return self

    def get(self):
        return self.min


class RollingMin(base.RollingUnivariate, utils.SortedWindow):
    """Running min over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> rolling_min = stats.RollingMin(2)
            >>> for x in X:
            ...     print(rolling_min.update(x).get())
            1
            -4
            -4
            -2
            -2
            1

    """

    def __init__(self, window_size):
        super().__init__(size=window_size)

    @property
    def window_size(self):
        return self.size

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        return self[0]
