from . import base
from . import mean
from . import summing


class Cov(base.Bivariate):
    """Covariance.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Examples
    --------

    >>> from river import stats

    >>> x = [-2.1,  -1,  4.3]
    >>> y = [   3, 1.1, 0.12]

    >>> cov = stats.Cov()

    >>> for xi, yi in zip(x, y):
    ...     print(cov.update(xi, yi).get())
    0.0
    -1.044999
    -4.286

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean_x = mean.Mean()
        self.mean_y = mean.Mean()
        self.cov = 0

    def update(self, x, y):
        dx = x - self.mean_x.get()
        self.mean_x.update(x)
        self.mean_y.update(y)
        dy = y - self.mean_y.get()
        self.cov += (dx * dy - self.cov) / max(1, self.mean_x.n - self.ddof)
        return self

    def get(self):
        return self.cov


class RollingCov(base.Bivariate):
    """Rolling covariance.

    Parameters
    ----------
    window_size
        Size of the window over which to compute the covariance.
    ddof
        Delta Degrees of Freedom.

    Here is the derivation, where $C$ denotes the covariance and $d$ is the amount of degrees of
    freedom:

    $$C = \\frac{1}{n - d} \\sum_{i=1}^n (x_i - \\bar{x}) (y_i - \\bar{y})$$

    $$C = \\frac{1}{n - d} \\sum_{i=1}^n x_i y_i - x_i \\bar{y} - \\bar{x} y_i + \\bar{x} \\bar{y}$$

    $$C = \\frac{1}{n - d} (\\sum_{i=1}^n x_i y_i - \\bar{y} \\sum_{i=1}^n x_i - \\bar{x} \\sum_{i=1}^n y_i + \\sum_{i=1}^n \\bar{x}\\bar{y})$$

    $$C = \\frac{1}{n - d} (\\sum_{i=1}^n x_i y_i - \\bar{y} n \\bar{x} - \\bar{x} n \\bar{y} + n \\bar{x}\\bar{y})$$

    $$C = \\frac{1}{n - d} (\\sum_{i=1}^n x_i y_i - n \\bar{x} \\bar{y})$$

    $$C = \\frac{1}{n - d} (\\sum_{i=1}^n x_i y_i - \\frac{\\sum_{i=1}^n x_i \\sum_{i=1}^n y_i}{n})$$

    The derivation is straightforward and somewhat trivial, but is a nice example of reformulating
    an equation so that it can be updated online. Note that we cannot apply this derivation to the
    non-rolling version of covariance because that would result in sums that grow infinitely, which
    can potentially cause numeric overflow.

    Examples
    --------

    >>> from river import stats

    >>> x = [-2.1,  -1, 4.3, 1, -2.1,  -1, 4.3]
    >>> y = [   3, 1.1, .12, 1,    3, 1.1, .12]

    >>> rcov = stats.RollingCov(3)

    >>> for xi, yi in zip(x, y):
    ...     print(rcov.update(xi, yi).get())
    0.0
    -1.045
    -4.286
    -1.382
    -4.589
    -1.415
    -4.286

    """

    def __init__(self, window_size, ddof=1):
        self.ddof = ddof
        self.sx = summing.RollingSum(window_size)
        self.sy = summing.RollingSum(window_size)
        self.sxy = summing.RollingSum(window_size)

    @property
    def window_size(self):
        return self.sxy.window_size

    def update(self, x, y):
        self.sx.update(x)
        self.sy.update(y)
        self.sxy.update(x * y)
        return self

    def get(self):
        n = len(self.sx)  # current window size
        return (self.sxy.get() - self.sx.get() * self.sy.get() / n) / max(
            1, n - self.ddof
        )
