import copy

import numpy as np

from river import stats


class Var(stats.base.Univariate):
    """Running variance using Welford's algorithm.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n`
        represents the number of seen elements.

    Attributes
    ----------
    mean
        It is necessary to calculate the mean of the data in order to calculate its variance.

    Notes
    -----
    The outcomes of the incremental and parallel updates are consistent with numpy's
    batch processing when $\\text{ddof} \\le 1$.

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
    [^2]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = stats.Mean()
        self._S = 0

    def update(self, x, w=1.0):
        mean_old = self.mean.get()
        self.mean.update(x, w)
        mean_new = self.mean.get()
        self._S += w * (x - mean_old) * (x - mean_new)
        return self

    def revert(self, x, w=1.0):
        mean_old = self.mean.get()
        self.mean.revert(x, w)
        mean_new = self.mean.get()
        self._S -= w * (x - mean_old) * (x - mean_new)
        return self

    def update_many(self, X: np.ndarray):
        mean_old = self.mean.get()
        self.mean.update_many(X)
        mean_new = self.mean.get()
        self._S += np.sum(
            np.multiply(np.subtract(X, mean_old), np.subtract(X, mean_new))
        )
        return self

    def get(self):
        if self.mean.n > self.ddof:
            return self._S / (self.mean.n - self.ddof)
        return 0.0

    @classmethod
    def _from_state(cls, n, m, sig, *, ddof=1):
        new = cls(ddof=ddof)
        new.mean = stats.Mean._from_state(n, m)  # noqa
        # scale the second order statistic
        new._S = (n - ddof) * sig

        return new

    def __iadd__(self, other):

        S = (
            self._S
            + other._S
            + (self.mean.get() - other.mean.get()) ** 2
            * self.mean.n
            * other.mean.n
            / (self.mean.n + other.mean.n)
        )
        self.mean += other.mean
        self._S = S

        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __isub__(self, other):

        self.mean -= other.mean

        S = (
            self._S
            - other._S
            - (self.mean.get() - other.mean.get()) ** 2
            * self.mean.n
            * other.mean.n
            / (self.mean.n + other.mean.n)
        )
        self._S = S

        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result


class RollingVar(stats.base.RollingUnivariate):
    """Running variance over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n`
        represents the number of seen elements.

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
        self._sos = 0
        self._rolling_mean = stats.RollingMean(window_size=window_size)

    @property
    def window_size(self):
        return self._rolling_mean.window_size

    def update(self, x):
        if len(self._rolling_mean.window) >= self._rolling_mean.window_size:
            self._sos -= self._rolling_mean.window[0] ** 2

        self._sos += x * x
        self._rolling_mean.update(x)
        return self

    @property
    def correction_factor(self):
        n = len(self._rolling_mean.window)
        if n > self.ddof:
            return n / (n - self.ddof)
        return 1

    def get(self):
        try:
            var = (
                self._sos / len(self._rolling_mean.window)
            ) - self._rolling_mean.get() ** 2
            return self.correction_factor * var
        except ZeroDivisionError:
            return 0.0
