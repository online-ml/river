import collections

from . import base


__all__ = ['Multinomial']


class Multinomial(collections.Counter, base.DiscreteDistribution):
    """Multinomial distribution for categorical data.

    Example:

        ::

            >>> from creme import proba

            >>> p = proba.Multinomial(['green'] * 3)
            >>> p = p.update('red')

            >>> p.pmf('red')
            0.25

            >>> p.update('red').update('red').pmf('green')
            0.5

    """

    def __init__(self, events=None):
        super().update(events)
        self._n = sum(self.values())

    @property
    def n_samples(self):
        return self._n

    def update(self, x):
        super().update([x])
        self._n += 1
        return self

    def pmf(self, x):
        try:
            return self[x] / self._n
        except ZeroDivisionError:
            return 0.

    def __str__(self):
        return '\n'.join(f'P({c}) = {self.pmf(c):.3f}' for c in self)
