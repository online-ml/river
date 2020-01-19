import bisect
import statistics
import collections

from .. import utils

from . import base


__all__ = [
    'Median',
    'RollingMedian'
]


class Median(base.Univariate):
    """Running median.

    The mode is simply the most common value. An approximate mode can be computed by setting the
    number of first unique values to count.

    Attributes:
        sorted_list (list): list sorted in ascending order.
        len_ (int): length of sorted list.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, 4, 2, -1, 0, 4, 2]
            >>> median = stats.Median()
            >>> for x in X:
            ...     print(median.update(x).get())
            1
            2.5
            2
            1.5
            1
            1.5
            2
    """

    def __init__(self):
        self.sorted_list = []
        self.len_ = 0

    @property
    def name(self):
        return 'median'

    def update(self, x):
        bisect.insort(self.sorted_list, x)
        self.len_ += 1
        return self

    def get(self):
        if self.len_ % 2:
            return self.sorted_list[self.len_//2]
        else:
            return (self.sorted_list[self.len_ // 2 - 1] + self.sorted_list[self.len_ // 2]) / 2


class RollingMedian(base.RollingUnivariate, utils.SortedWindow):
    """Running median over a window.

    The mode is the most common value.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, 4, 2, -5, 0, -2, 6]
            >>> rolling_median = stats.RollingMedian(window_size=3)
            >>> for x in X:
            ...     print(rolling_median.update(x).get())
            1
            2.5
            2
            2
            0
            -2
            0
    """

    def __init__(self, window_size):
        super().__init__(size=window_size)

    @property
    def window_size(self):
        return self.size

    def update(self, x):
        super().append(x)
        return self

    def get(self):
        if len(self) % 2:
            return self[len(self)//2]
        else:
            return (self[len(self) // 2 - 1] + self[len(self) // 2]) / 2
