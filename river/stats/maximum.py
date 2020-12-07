import math

from river import utils

from . import base


class Max(base.Univariate):
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


class RollingMax(base.RollingUnivariate, utils.SortedWindow):
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
        super().__init__(size=window_size)

    @property
    def window_size(self):
        return self.size

    def update(self, x):
        self.append(x)
        return self

    def get(self):
        return self[-1]


class AbsMax(base.Univariate):
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


class RollingAbsMax(base.RollingUnivariate, utils.SortedWindow):
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
        super().__init__(size=window_size)

    @property
    def window_size(self):
        return self.size

    def update(self, x):
        self.append(abs(x))
        return self

    def get(self):
        return self[-1]
