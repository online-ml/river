cimport river.stats.base
cimport river.stats.mean

from . import base
from . import mean


cdef class Var(river.stats.base.Univariate):
    """Running variance using Welford's algorithm.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n`
        represents the number of seen elements.

    Attributes
    ----------
    mean : stats.Mean
        The running mean.
    sos : float
        The running sum of squares.

    Examples
    --------

    >>> import river.stats

    >>> X = [3, 5, 4, 7, 10, 12]

    >>> var = river.stats.Var()
    >>> for x in X:
    ...     print(var.update(x).get())
    0.0
    2.0
    1.0
    2.916666
    7.7
    12.56666

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    cdef readonly long ddof
    cdef readonly river.stats.mean.Mean mean
    cdef readonly double sigma

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = mean.Mean()

    cpdef Var update(self, double x, double w=1.):
        mean = self.mean.get()
        self.mean.update(x, w)
        if self.mean.n > self.ddof:
            self.sigma += w * ((x - mean) * (x - self.mean.get()) - self.sigma) / (self.mean.n - self.ddof)
        return self

    cpdef double get(self):
        return self.sigma


class RollingVar(base.RollingUnivariate):
    """Running variance over a window.

    Parameters
    ----------
    window_size : int
        Size of the rolling window.
    ddof : int
        Delta Degrees of Freedom for the variance.

    Attributes
    ----------
    sos : float
        Sum of squares over the current window.
    rmean : stats.RollingMean

    Examples
    --------

    >>> import river

    >>> X = [1, 4, 2, -4, -8, 0]

    >>> rvar = river.stats.RollingVar(ddof=1, window_size=2)
    >>> for x in X:
    ...     print(rvar.update(x).get())
    0.0
    4.5
    2.0
    18.0
    8.0
    32.0

    >>> rvar = river.stats.RollingVar(ddof=1, window_size=3)
    >>> for x in X:
    ...     print(rvar.update(x).get())
    0.0
    4.5
    2.333333
    17.333333
    25.333333
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
            var = (self.sos / len(self.rolling_mean)) - self.rolling_mean.get() ** 2
            return self.correction_factor * var
        except ZeroDivisionError:
            return 0.
