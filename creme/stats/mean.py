from . import base


class Mean(base.RunningStatistic):
    """Running mean.

    Attributes:
        count (stats.Count)
        mean (float)

    Example:

    ::

        >>> from creme import stats

        >>> X = [-5, -3, -1, 1, 3, 5]
        >>> mean = stats.Mean()
        >>> for x in X:
        ...     print(mean.update(x).get())
        -5.0
        -4.0
        -3.0
        -2.0
        -1.0
        0.0

    """

    def __init__(self):
        self.n = 0
        self.mean = 0

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        self.n += 1
        self.mean += (x - self.mean) / self.n if self.n > 0 else 0
        return self

    def get(self):
        return self.mean
