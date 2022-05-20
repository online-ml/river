import collections

from river import stats


class Sum(stats.base.Univariate):
    """Running sum.

    Attributes
    ----------
    sum : float
        The running sum.

    Examples
    --------

    >>> from river import stats

    >>> X = [-5, -3, -1, 1, 3, 5]
    >>> mean = stats.Sum()
    >>> for x in X:
    ...     print(mean.update(x).get())
    -5.0
    -8.0
    -9.0
    -8.0
    -5.0
    0.0

    """

    def __init__(self):
        self.sum = 0.0

    def update(self, x):
        self.sum += x
        return self

    def get(self):
        return self.sum


class RollingSum(stats.base.RollingUnivariate):
    """Running sum over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Attributes
    ----------
    sum : int
        The running rolling sum.

    Examples
    --------

    >>> from river import stats

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

    def __init__(self, window_size: int):
        self.window = collections.deque(maxlen=window_size)
        self.sum = 0

    @property
    def window_size(self):
        return self.window.maxlen

    def update(self, x):
        if len(self.window) == self.window_size:
            self.sum -= self.window[0]
        self.sum += x
        self.window.append(x)
        return self

    def get(self):
        return self.sum
