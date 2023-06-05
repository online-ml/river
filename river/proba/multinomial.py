from __future__ import annotations

import collections
import typing

from river.proba import base

__all__ = ["Multinomial"]


class Multinomial(base.DiscreteDistribution):
    """Multinomial distribution for categorical data.

    Parameters
    ----------
    events
        An optional list of events that already occurred.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import proba

    >>> p = proba.Multinomial(['green'] * 3)
    >>> p = p.update('red')

    >>> p('red')
    0.25

    >>> p = p.update('red').update('red')
    >>> p('green')
    0.5

    >>> p = p.revert('red').revert('red')
    >>> p('red')
    0.25

    You can wrap this with a `utils.Rolling` to measure a distribution over a window:

    >>> from river import utils

    >>> X = ['red', 'green', 'green', 'blue', 'blue']

    >>> dist = utils.Rolling(
    ...     proba.Multinomial(),
    ...     window_size=3
    ... )

    >>> for x in X:
    ...     dist = dist.update(x)
    ...     print(dist)
    ...     print()
    P(red) = 1.000
    <BLANKLINE>
    P(red) = 0.500
    P(green) = 0.500
    <BLANKLINE>
    P(green) = 0.667
    P(red) = 0.333
    <BLANKLINE>
    P(green) = 0.667
    P(blue) = 0.333
    P(red) = 0.000
    <BLANKLINE>
    P(blue) = 0.667
    P(green) = 0.333
    P(red) = 0.000
    <BLANKLINE>

    You can wrap this with a `utils.Rolling` to measure a distribution over a window of time:

    >>> import datetime as dt

    >>> X = ['red', 'green', 'green', 'blue']
    >>> days = [1, 2, 3, 4]

    >>> dist = utils.TimeRolling(
    ...     proba.Multinomial(),
    ...     period=dt.timedelta(days=2)
    ... )

    >>> for x, day in zip(X, days):
    ...     dist = dist.update(x, t=dt.datetime(2019, 1, day))
    ...     print(dist)
    ...     print()
    P(red) = 1.000
    <BLANKLINE>
    P(red) = 0.500
    P(green) = 0.500
    <BLANKLINE>
    P(green) = 1.000
    P(red) = 0.000
    <BLANKLINE>
    P(green) = 0.500
    P(blue) = 0.500
    P(red) = 0.000
    <BLANKLINE>

    """

    def __init__(self, events: dict | list | None = None, seed=None):
        super().__init__(seed)
        self.events = events
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

    def revert(self, x):
        self.counts.subtract([x])
        self._n -= 1
        return self

    def sample(self):
        return self._rng.choices(list(self.counts.keys()), weights=list(self.counts.values()))[0]

    def __call__(self, x):
        try:
            return self.counts[x] / self._n
        except ZeroDivisionError:
            return 0.0

    def __repr__(self):
        return "\n".join(f"P({c}) = {self(c):.3f}" for c, _ in self.counts.most_common())
