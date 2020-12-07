from . import base
from . import count


class CentralMoments(base.Univariate):
    """Computes central moments using Welford's algorithm.

    Attributes
    ----------
    count : stats.Count
    delta: float
    sum_delta : float
        Mean of sum of differences.
    M1 : float
        Sums of powers of differences from the mean order 1.
    M2 : float)
        Sums of powers of differences from the mean order 2.
    M3 : float
        Sums of powers of differences from the mean order 3.
    M4 : float
        Sums of powers of differences from the mean order 4.

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def __init__(self):
        self.count = count.Count()

        self.delta = 0
        self.sum_delta = 0

        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M4 = 0

    def _update_delta(self, x):
        self.delta = (x - self.sum_delta) / self.count.get()
        return self

    def _update_sum_delta(self):
        self.sum_delta += self.delta
        return self

    def _update_m1(self, x):
        self.M1 = (x - self.sum_delta) * self.delta * (self.count.get() - 1)
        return self

    def _update_m2(self):
        self.M2 += self.M1
        return self

    def _update_m3(self):
        self.M3 += self.M1 * self.delta * (self.count.get() - 2) - 3 * self.delta * self.M2
        return self

    def _update_m4(self):
        delta_square = self.delta ** 2
        self.M4 += (
            self.M1 * delta_square * (self.count.get() ** 2 - 3 * self.count.get() + 3)
            + 6 * delta_square * self.M2
            - 4 * self.delta * self.M3
        )
        return self
