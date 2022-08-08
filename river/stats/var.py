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

    >>> from river import stats

    >>> X = [3, 5, 4, 7, 10, 12]

    >>> var = stats.Var()
    >>> for x in X:
    ...     print(var.update(x).get())
    0.0
    2.0
    1.0
    2.916666
    7.7
    12.56666

    You can measure a rolling variance by using a `utils.Rolling` wrapper:

    >>> from river import utils

    >>> X = [1, 4, 2, -4, -8, 0]
    >>> rvar = utils.Rolling(stats.Var(ddof=1), window_size=3)
    >>> for x in X:
    ...     print(rvar.update(x).get())
    0.0
    4.5
    2.333333
    17.333333
    25.333333
    16.0

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
    [^2]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)
    [^3]: Schubert, E. and Gertz, M., 2018, July. Numerically stable parallel computation of
    (co-)variance. In Proceedings of the 30th International Conference on Scientific and
    Statistical Database Management (pp. 1-12).

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = stats.Mean()
        self._S = 0

    @property
    def n(self):
        return self.mean.n

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
        self._S += np.sum(np.multiply(np.subtract(X, mean_old), np.subtract(X, mean_new)))
        return self

    def get(self):
        if self.n > self.ddof:
            return self._S / (self.n - self.ddof)
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
            + (self.mean.get() - other.mean.get()) ** 2 * self.n * other.n / (self.n + other.n)
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
            * self.n
            * other.mean.n
            / (self.n + other.mean.n)
        )
        self._S = S

        return self

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result
