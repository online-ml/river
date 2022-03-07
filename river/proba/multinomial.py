import collections
import typing

from . import base

__all__ = ["Multinomial"]


class Multinomial(base.DiscreteDistribution):
    """Multinomial distribution for categorical data.

    Parameters
    ----------
    events
        An optional list of events that already occurred.

    Examples
    --------

    >>> from river import proba

    >>> p = proba.Multinomial(['green'] * 3)
    >>> p = p.update('red')

    >>> p.pmf('red')
    0.25

    >>> p.update('red').update('red').pmf('green')
    0.5

    """

    def __init__(self, events: typing.Union[dict, list] = None):
        self.counts: typing.Counter[typing.Any] = collections.Counter(events)
        self._n = sum(self.counts.values())

    @property
    def n_samples(self):
        return self._n

    def __iter__(self):
        return iter(self.counts)

    def __len__(self):
        return len(self.counts)

    @property
    def mode(self):
        return self.counts.most_common(1)[0][0]

    def update(self, x):
        self.counts.update([x])
        self._n += 1
        return self

    def pmf(self, x):
        try:
            return self.counts[x] / self._n
        except ZeroDivisionError:
            return 0.0

    def __str__(self):
        return "\n".join(
            f"P({c}) = {self.pmf(c):.3f}" for c in self.counts.most_common()
        )
