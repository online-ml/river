from . import base, cov, var


class PearsonCorr(base.Bivariate):
    """Online Pearson correlation.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom.

    Attributes
    ----------
    var_x : stats.Var
        Running variance of `x`.
    var_y : stats.Var
        Running variance of `y`.
    cov_xy : stats.Cov
        Running covariance of `x` and `y`.

    Examples
    --------

    >>> from river import stats

    >>> x = [0, 0, 0, 1, 1, 1, 1]
    >>> y = [0, 1, 2, 3, 4, 5, 6]

    >>> pearson = stats.PearsonCorr()

    >>> for xi, yi in zip(x, y):
    ...     print(pearson.update(xi, yi).get())
    0
    0
    0
    0.774596
    0.866025
    0.878310
    0.866025

    """

    def __init__(self, ddof=1):
        self.var_x = var.Var(ddof=ddof)
        self.var_y = var.Var(ddof=ddof)
        self.cov_xy = cov.Cov(ddof=ddof)

    @property
    def ddof(self):
        return self.cov_xy.ddof

    def update(self, x, y):
        self.var_x.update(x)
        self.var_y.update(y)
        self.cov_xy.update(x, y)
        return self

    def get(self):
        var_x = self.var_x.get()
        var_y = self.var_y.get()
        if var_x and var_y:
            return self.cov_xy.get() / (var_x * var_y) ** 0.5
        return 0


class RollingPearsonCorr(base.Bivariate):
    """Rolling Pearson correlation.

    Parameters
    ----------
    window_size
        Amount of samples over which to compute the correlation.
    ddof
        Delta Degrees of Freedom.

    Attributes
    ----------
    var_x : stats.Var
        Running variance of `x`.
    var_y : stats.Var
        Running variance of `y`.
    cov_xy : stats.Cov
        Running covariance of `x` and `y`.

    Examples
    --------

    >>> from river import stats

    >>> x = [0, 0, 0, 1, 1, 1, 1]
    >>> y = [0, 1, 2, 3, 4, 5, 6]

    >>> pearson = stats.RollingPearsonCorr(window_size=4)

    >>> for xi, yi in zip(x, y):
    ...     print(pearson.update(xi, yi).get())
    0
    0
    0
    0.7745966692414834
    0.894427190999916
    0.7745966692414834
    0

    """

    def __init__(self, window_size, ddof=1):
        self.var_x = var.RollingVar(window_size=window_size, ddof=ddof)
        self.var_y = var.RollingVar(window_size=window_size, ddof=ddof)
        self.cov_xy = cov.RollingCov(window_size=window_size, ddof=ddof)

    @property
    def window_size(self):
        return self.cov_xy.window_size

    def update(self, x, y):
        self.var_x.update(x)
        self.var_y.update(y)
        self.cov_xy.update(x, y)
        return self

    def get(self):
        var_x = self.var_x.get()
        var_y = self.var_y.get()
        if var_x and var_y:
            return self.cov_xy.get() / (var_x * var_y) ** 0.5
        return 0
