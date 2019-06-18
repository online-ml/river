import collections

from . import base


__all__ = ['Multinomial']


class Multinomial(collections.Counter, base.DiscreteDistribution):
    """Normal distribution with parameters mu and sigma.

    Example:

        ::

            >>> from creme import proba

            >>> p = proba.Multinomial()
            >>> p = p.update('red')

            >>> p.pmf('red')
            1.0

            >>> p.update(6).update(1).update(6).pmf('red')
            0.25

    """

    def __init__(self):
        self.total = 0

    def update(self, x):
        super().update([x])
        self.total += 1
        return self

    def pmf(self, x):
        try:
            return self[x] / self.total
        except ZeroDivisionError:
            return 0.
