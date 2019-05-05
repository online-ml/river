from . import variance


class SEM(variance.Variance):
    """Running standard error of the mean using Welford's algorithm.

    Parameters:
        ddof (int): Delta Degrees of Freedom. The divisor used in calculations is $n$ - ddof,
            where $n$ represents the number of seen elements.

    Attributes:
        n (int): Number of observations.

    Example:

        ::

            >>> import creme.stats

            >>> X = [3, 5, 4, 7, 10, 12]

            >>> sem = creme.stats.SEM()
            >>> for x in X:
            ...     print(sem.update(x).get())
            0.0
            1.0
            0.577350...
            0.853912...
            1.240967...
            1.447219...

    References:

        1. `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    @property
    def name(self):
        return 'sem'

    def get(self):
        return (super().get() / self.mean.n) ** 0.5
