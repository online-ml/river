from . import base


class Sum(base.Univariate):
    """Running sum.

    Attributes:
        sum (float) : The running sum.

    Example:

        ::

            >>> from creme import stats

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
        self.sum = 0.

    @property
    def name(self):
        return 'sum'

    def update(self, x):
        self.sum += x
        return self

    def get(self):
        return self.sum
