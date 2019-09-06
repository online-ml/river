import collections

from . import base


__all__ = ['Multinomial']


class Multinomial(collections.Counter, base.DiscreteDistribution):
    """Multinomial distribution for categorical data.

    Example:

        ::

            >>> from creme import proba

            >>> p = proba.Multinomial({'green': 3})
            >>> p = p.update('red')

            >>> p.pmf('red')
            0.25

            >>> p.update('red').update('red').pmf('green')
            0.5

    """

    def __init__(self, initial_counts=None):
        self.n = 0
        if initial_counts is not None:
            for label, count in initial_counts.items():
                self[label] += count
                self.n += count

    def update(self, x):
        super().update([x])
        self.n += 1
        return self

    def pmf(self, x):
        try:
            return self[x] / self.n
        except ZeroDivisionError:
            return 0.
