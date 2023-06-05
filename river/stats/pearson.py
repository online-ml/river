from __future__ import annotations

from river import stats


class PearsonCorr(stats.base.Bivariate):
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

    You can also do this in a rolling fashion:

    >>> from river import utils

    >>> x = [0, 0, 0, 1, 1, 1, 1]
    >>> y = [0, 1, 2, 3, 4, 5, 6]

    >>> pearson = utils.Rolling(stats.PearsonCorr(), window_size=4)

    >>> for xi, yi in zip(x, y):
    ...     print(pearson.update(xi, yi).get())
    0
    0
    0
    0.7745966692414834
    0.8944271909999159
    0.7745966692414832
    -4.712160915387242e-09

    """

    def __init__(self, ddof=1):
        self.var_x = stats.Var(ddof=ddof)
        self.var_y = stats.Var(ddof=ddof)
        self.cov_xy = stats.Cov(ddof=ddof)

    @property
    def ddof(self):
        return self.cov_xy.ddof

    def update(self, x, y):
        self.var_x.update(x)
        self.var_y.update(y)
        self.cov_xy.update(x, y)
        return self

    def revert(self, x, y):
        self.var_x.revert(x)
        self.var_y.revert(y)
        self.cov_xy.revert(x, y)
        return self

    def get(self):
        var_x = self.var_x.get()
        var_y = self.var_y.get()
        if var_x and var_y:
            return self.cov_xy.get() / (var_x * var_y) ** 0.5
        return 0
