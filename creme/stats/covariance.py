from . import base


class Covariance(base.Bivariate):
    """Running covariance.

    Parameters:
        ddof (int): Delta Degrees of Freedom.

    Example:

        >>> from creme import stats

        >>> x = [-2.1,  -1,  4.3]
        >>> y = [   3, 1.1, 0.12]

        >>> cov = stats.Covariance()

        >>> for xi, yi in zip(x, y):
        ...     print(cov.update(xi, yi).get())
        0.0
        -1.044999...
        -4.286

    References:

        1. `Algorithms for calculating variance <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance>`_

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean_x = 0
        self.mean_y = 0
        self.n = 0
        self.C = 0

    def update(self, x, y):
        self.n += 1
        dx = x - self.mean_x
        self.mean_x += dx / self.n
        self.mean_y += (y - self.mean_y) / self.n
        self.C += dx * (y - self.mean_y)
        return self

    def get(self):
        return self.C / max(1, self.n - self.ddof)
