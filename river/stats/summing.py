from __future__ import annotations

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

    >>> from river import utils

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> rolling_sum = utils.Rolling(stats.Sum(), window_size=2)
    >>> for x in X:
    ...     print(rolling_sum.update(x).get())
    1.0
    -3.0
    -1.0
    1.0
    0.0
    3.0

    """

    def __init__(self):
        self.sum = 0.0

    def update(self, x):
        self.sum += x
        return self

    def revert(self, x):
        self.sum -= x
        return self

    def get(self):
        return self.sum
