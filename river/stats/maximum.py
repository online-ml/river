from __future__ import annotations

import math

from river import stats, utils


class Max(stats.base.Univariate):
    """Running max.

    Attributes
    ----------
    max : float
        The current max.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 5, -6]
    >>> _max = stats.Max()
    >>> for x in X:
    ...     print(_max.update(x).get())
    1
    1
    3
    3
    5
    5

    """

    def __init__(self):
        self.max = -math.inf

    def update(self, x):
        if x > self.max:
            self.max = x
        return self

    def get(self):
        return self.max


class RollingMax(stats.base.RollingUnivariate):
    """Running max over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------

    >>> from river import stats

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
            return self.window[-1]
        except IndexError:
            return None


class AbsMax(stats.base.Univariate):
    """Running absolute max.

    Attributes
    ----------
    abs_max : float
        The current absolute max.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 5, -6]
    >>> abs_max = stats.AbsMax()
    >>> for x in X:
    ...     print(abs_max.update(x).get())
    1
    4
    4
    4
    5
    6

    """

    def __init__(self):
        self.abs_max = 0.0

    def update(self, x):
        if abs(x) > self.abs_max:
            self.abs_max = abs(x)
        return self

    def get(self):
        return self.abs_max


class RollingAbsMax(stats.base.RollingUnivariate):
    """Running absolute max over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> rolling_absmax = stats.RollingAbsMax(window_size=2)
    >>> for x in X:
    ...     print(rolling_absmax.update(x).get())
    1
    4
    4
    3
    2
    2

    """

    def __init__(self, window_size: int):
        self.window = utils.SortedWindow(size=window_size)

    @property
    def window_size(self):
        return self.window.size

    def update(self, x):
        self.window.append(abs(x))
        return self

    def get(self):
        try:
            return self.window[-1]
        except IndexError:
            return None
