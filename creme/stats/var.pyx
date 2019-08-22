cimport base
cimport mean

from . import base
from . import mean


cdef class Var(base.Univariate):
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

            >>> var = creme.stats.Var()
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

    cdef readonly long ddof
    cdef readonly mean.Mean mean
    cdef readonly double sos

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = mean.Mean()
        self.sos = 0.

    cpdef Var update(self, double x):
        mean = self.mean.get()
        self.mean.update(x)
        self.sos += (x - mean) * (x - self.mean.get())
        return self

    cpdef double get(self):
        if self.mean.n > self.ddof:
            return self.sos / (self.mean.n - self.ddof)
        return 0.


class RollingVar(base.RollingUnivariate):
    """Running variance over a window.

    Parameters:
        window_size (int): Size of the rolling window.
        ddof (int): Delta Degrees of Freedom for the variance.

    Attributes:
        sos (float): Sum of squares.
        rolling_mean (stats.RollingMean): The running rolling mean.

    Example:

        ::

            >>> import creme

            >>> X = [1, 4, 2, -4, -8, 0]

            >>> rolling_variance = creme.stats.RollingVar(ddof=1, window_size=2)
            >>> for x in X:
            ...     print(rolling_variance.update(x).get())
            0.0
            4.5
            2.0
            18.0
            8.0
            32.0

            >>> rolling_variance = creme.stats.RollingVar(ddof=1, window_size=3)
            >>> for x in X:
            ...     print(rolling_variance.update(x).get())
            0.0
            4.5
            2.333333...
            17.333333...
            25.333333...
            16.0

    """

    def __init__(self, window_size, ddof=1):
        self.ddof = ddof
        self.sos = 0
        self.rolling_mean = mean.RollingMean(window_size=window_size)

    @property
    def window_size(self):
        return self.rolling_mean.window_size

    def update(self, x):
        if len(self.rolling_mean) >= self.rolling_mean.size:
            self.sos -= self.rolling_mean[0] ** 2

        self.sos += x * x
        self.rolling_mean.update(x)
        return self

    @property
    def correction_factor(self):
        n = len(self.rolling_mean)
        if n > self.ddof:
            return n / (n - self.ddof)
        return 1

    def get(self):
        try:
            variance = (self.sos / len(self.rolling_mean)) - self.rolling_mean.get() ** 2
            return self.correction_factor * variance
        except ZeroDivisionError:
            return 0.
