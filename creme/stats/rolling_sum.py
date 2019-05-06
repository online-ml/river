from .. import utils

from . import base


class RollingSum(base.Univariate, utils.Window):
    """Running sum over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Attributes:
        sum (int): The running rolling sum.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 1]
            >>> rolling_sum = stats.RollingSum(2)
            >>> for x in X:
            ...     print(rolling_sum.update(x).get())
            1
            -3
            -1
            1
            0
            3

    """

    def __init__(self, window_size):
        super().__init__(window_size)
        self.sum = 0

    @property
    def name(self):
        return f'rolling_{self.window_size}_sum'

    def update(self, x):
        if len(self) == self.window_size:
            self.sum -= self[0]
        self.sum += x
        self.append(x)
        return self

    def get(self):
        return self.sum
