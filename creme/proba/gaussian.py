import math

from .. import stats

from . import base


class Gaussian(base.Distribution):
    """Normal distribution with parameters mu and sigma.

    Example:

        ::

            >>> from creme import proba

            >>> p1 = proba.Gaussian().update(5)
            >>> p1.mode()
            5.0

            >>> p2 = proba.Gaussian().update(6).update(7)
            >>> p2.mode()
            6.5

            >>> p2.proba_of(6.5)
            0.564189...

    """

    def __init__(self):
        self.variance = stats.Variance()

    @property
    def mu(self):
        return self.variance.mean.get()

    @property
    def sigma(self):
        return self.variance.get() ** 0.5

    def update(self, x):
        self.variance.update(x)
        return self

    def mode(self):
        return self.mu

    def proba_of(self, x):
        variance = self.variance.get()

        if variance == 0:
            return 0

        return math.exp((x - self.mu) ** 2 / (-2 * variance)) / math.sqrt(math.tau * variance)
