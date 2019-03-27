from . import base
from . import mean


class Sem(base.RunningStatistic):
    """Running standard error of the mean using Welford's algorithm.

    Parameters:
        ddof (int): Delta Degrees of Freedom. The divisor used in calculations is $n$ - ddof,
            where $n$ represents the number of seen elements.

    Attributes:
        mean (stats.Mean)
        sos (float): The running sum of squares.

    Example:

        >>> import creme.stats

        >>> X = [3, 5, 4, 7, 10, 12]

        >>> sem = creme.stats.Sem()
        >>> for x in X:
        ...     print(sem.update(x).get())
        0.0
        1.0
        0.577350...
        0.853912...
        1.240967...
        1.447219...

    References:

    - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = mean.Mean()
        self.sos = 0
        self.n = 0

    @property
    def name(self):
        return 'Sem'

    def update(self, x):
        self.sos += (x - self.mean.get()) * (x - self.mean.update(x).get())
        self.n += 1
        return self

    def get(self):
        return (self.sos / max(1, self.mean.n - self.ddof))**0.5 / (self.n ** 0.5)
