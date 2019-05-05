from . import base
from . import mean


class Variance(base.Univariate):
    """Running variance using Welford's algorithm.

    Parameters:
        ddof (int): Delta Degrees of Freedom. The divisor used in calculations is $n$ - ddof,
            where $n$ represents the number of seen elements.

    Attributes:
        mean (stats.Mean): The running mean.
        sos (float): The running sum of squares.

    Example:

        ::

            >>> import creme.stats

            >>> X = [3, 5, 4, 7, 10, 12]

            >>> var = creme.stats.Variance()
            >>> for x in X:
            ...     print(var.update(x).get())
            0.0
            2.0
            1.0
            2.916666...
            7.7
            12.56666...

    References:

        1. `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = mean.Mean()
        self.sos = 0

    @property
    def name(self):
        return 'variance'

    @property
    def n(self):
        return self.mean.n

    def update(self, x):
        self.sos += (x - self.mean.get()) * (x - self.mean.update(x).get())
        return self

    def get(self):
        return self.sos / max(1, self.mean.n - self.ddof)
