import collections

from . import base


class Multinomial(base.DiscreteDistribution):
    """Multinomial distribution.

    Example:

        >>> d1 = Multinomial().update('sunny')
        >>> d1.mode()
        'sunny'

        >>> d2 = Multinomial().update('rainy').update('rainy')
        >>> d2.mode()
        'rainy'

        >>> d1 += d2
        >>> d1.mode()
        'rainy'

        >>> d1.pmf('cloudy')
        0.0

    """

    def __init__(self):
        self.counter = collections.Counter()
        self.n = 0

    def update(self, x):
        self.counter.update([x])
        self.n += 1
        return self

    def mode(self):
        if self.counter:
            return max(self.counter, key=self.counter.get)
        return None

    def pmf(self, x):
        return self.counter.get(x, 0) / self.n

    def __iadd__(self, other):
        self.counter.update(other.counter)
        self.n += other.n
        return self

    def __imul__(self, constant):
        for i in self.counter:
            self.counter[i] *= constant
        return self

    def __copy__(self):
        """Custom copying to make ``copy.copy`` faster."""
        clone = Multinomial()
        clone.counter = collections.Counter(self.counter)
        clone.n = self.n
        return clone

    def __deepcopy__(self, memo):
        """Custom copying to make ``copy.deepcopy`` faster."""
        return self.__copy__()
