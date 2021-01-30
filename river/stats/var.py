import copy

from . import base, mean


class Var(base.Univariate):
    r"""Running variance using Welford's algorithm.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n`
        represents the number of seen elements.

    Attributes
    ----------
    mean : stats.Mean
        The running mean.
    sigma : float
        The running variance.

    Notes
    -----
    The outcomes of the incremental and parallel updates are consistent with numpy's
    batch processing when $\\text{ddof} \le 1$.

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
        self.mean = mean.Mean()
        self.sigma = 0.0

    def update(self, x, w=1.0):
        mean = self.mean.get()
        self.mean.update(x, w)
        if self.mean.n > self.ddof:
            self.sigma += (
                w
                * ((x - mean) * (x - self.mean.get()) - self.sigma)
                / (self.mean.n - self.ddof)
            )
        return self

    def get(self):
        return self.sigma

    def __iadd__(self, other):
        if other.mean.n <= self.ddof:
            return self

        old_n = self.mean.n
        delta = other.mean.get() - self.mean.get()

        self.mean += other.mean
        # scale and merge sigma
        self.sigma = (old_n - self.ddof) * self.sigma + (
            other.mean.n - other.ddof
        ) * other.sigma
        # apply correction
        self.sigma = (
            self.sigma + (delta * delta) * (old_n * other.mean.n) / self.mean.n
        ) / (self.mean.n - self.ddof)

        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other

        return result

    def __isub__(self, other):
        old_n = self.mean.n
        delta = 0.0

        self.mean -= other.mean

        if self.mean.n > 0 and self.mean.n > self.ddof:
            delta = other.mean.get() - self.mean.get()
            # scale both sigma and take the difference
            self.sigma = (old_n - self.ddof) * self.sigma - (
                other.mean.n - other.ddof
            ) * other.sigma
            # apply the correction
            self.sigma = (
                self.sigma - (delta * delta) * (self.mean.n * other.mean.n) / old_n
            ) / (self.mean.n - self.ddof)

        else:
            self.sigma = 0.0

        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other

        return result


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
            return 0.0
