import bisect
import statistics
import collections

from .. import utils

from . import base


__all__ = ['Median']


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
