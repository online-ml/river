from . import variance


class Sem(variance.Variance):
    """Running standard error of the mean using Welford's algorithm.

    Parameters:
        ddof (int): Delta Degrees of Freedom. The divisor used in calculations is $n$ - ddof,
            where $n$ represents the number of seen elements.

    Attributes:
        n (int): Number of observations.

    Example:

        >>> import creme.stats

        >>> X = [3, 5, 4, 7, 10, 12]

        >>> sem = creme.stats.Sem()
        >>> for x in X:
        ...     print(sem.update(x).get())
        0.0
        1.0
        0.577350...
        0.853912...
        1.240967...
        1.447219...

    References:

    - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    def __init__(self, ddof=1):
        super().__init__(
            ddof=ddof,
        )
        self.n = 0

    def update(self, x):
        super().update(x)
        self.n += 1
        return self

    @property
    def name(self):
        return 'SEM'

    def get(self):
        return (super().get()**0.5) / (self.n ** 0.5)
