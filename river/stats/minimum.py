from __future__ import annotations

import math

from river import stats, utils


class Min(stats.base.Univariate):
    """Running min.

    Attributes
    ----------
    min : float
        The current min.

    """

    def __init__(self):
        self.min = math.inf

    def update(self, x):
        if x < self.min:
            self.min = x
        return self

    def get(self):
        return self.min


class RollingMin(stats.base.RollingUnivariate):
    """Running min over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------

    >>> from river import stats

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

    def __init__(self, window_size: int):
        self.window = utils.SortedWindow(size=window_size)

    @property
    def window_size(self):
        return self.window.size

    def update(self, x):
        self.window.append(x)
        return self

    def get(self):
        try:
            return self.window[0]
        except IndexError:
            return None
