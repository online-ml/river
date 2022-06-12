from . import var


class SEM(var.Var):
    """Running standard error of the mean using Welford's algorithm.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n` is the
        number of seen elements.

    Attributes
    ----------
    n : int
        Number of observations.

    Examples
    --------

    >>> from river import stats

    >>> X = [3, 5, 4, 7, 10, 12]

    >>> sem = stats.SEM()
    >>> for x in X:
    ...     print(sem.update(x).get())
    0.0
    1.0
    0.577350
    0.853912
    1.240967
    1.447219

    >>> from river import utils

    >>> X = [1, 4, 2, -4, -8, 0]

    >>> rolling_sem = utils.Rolling(stats.SEM(ddof=1), window_size=3)
    >>> for x in X:
    ...     print(rolling_sem.update(x).get())
    0.0
    1.5
    0.881917
    2.403700
    2.905932
    2.309401

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def get(self):
        try:
            return (super().get() / self.mean.n) ** 0.5
        except ZeroDivisionError:
            return None
