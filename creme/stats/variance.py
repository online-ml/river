from . import base
from . import mean


class Variance(base.RunningStatistic):
    """Computes a running variance using Welford's algorithm.

    Attributes:
        mean (stats.Mean)
        sos (float): The running sum of squares.

    References:

    - `Welford's online algorithm <https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Welford's_Online_algorithm>`_

    """

    def __init__(self):
        self.mean = mean.Mean()
        self.sos = 0

    @property
    def name(self):
        return 'variance'

    def update(self, x):
        old_mean = self.mean.get()
        new_mean = self.mean.update(x).get()
        self.sos += (x - old_mean) * (x - new_mean)
        return self

    def get(self):
        return self.sos / self.mean.count.n if self.sos else 0
