import copy
import math

from .. import stats

from . import base


class Normal(base.ContinuousDistribution):
    """Normal distribution with parameters mu and sigma.

    Example:

        ::

            >>> d1 = Normal().update(5)
            >>> d1.mode()
            5.0

            >>> d2 = Normal().update(6).update(7)
            >>> d2.mode()
            6.5

            >>> d1 += d2
            >>> d1.mode()
            11.5

            >>> d2.pdf(6.5)
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

    def pdf(self, x):
        variance = self.variance.get()

        if variance == 0:
            return 0

        return math.exp((x - self.mu) ** 2 / (-2 * variance)) / math.sqrt(math.tau * variance)

    def __iadd__(self, other):
        if isinstance(other, Normal):
            other = other.variance.mean
        self.variance.mean += other
        return self

    def __imul__(self, constant):
        self.variance.mean *= constant
        return self

    def __copy__(self):
        """Custom copying to make the ``copy.copy`` method faster."""
        clone = Normal()
        clone.variance = copy.copy(self.variance)
        return clone

    def __deepcopy__(self, memo):
        return self.__copy__()
