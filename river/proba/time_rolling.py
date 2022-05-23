import bisect
import datetime as dt
import typing

from . import base


class TimeRolling(base.Distribution):
    """Wrapper for measuring probability distributions over a period of time.

    Parameters
    ----------
    dist
        A distribution.
    period
        A period of time.

    Examples
    --------

    >>> import datetime as dt
    >>> from river import proba

    >>> X = ['red', 'green', 'green', 'blue']
    >>> days = [1, 2, 3, 4]

    >>> dist = proba.TimeRolling(
    ...     proba.Multinomial(),
    ...     period=dt.timedelta(days=2)
    ... )

    >>> for x, day in zip(X, days):
    ...     dist = dist.update(x, dt.datetime(2019, 1, day))
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

    def __init__(self, dist: base.Distribution, period: dt.timedelta):
        self.dist = dist
        self.period = period
        self._events: typing.List[typing.Tuple[dt.datetime, typing.Any]] = []
        self._latest = dt.datetime(1, 1, 1)

    def update(self, x, t: dt.datetime):
        self.dist.update(x)
        bisect.insort_left(self._events, (t, x))

        # There will only be events to revert if the new event if younger than the previously seen
        # youngest event
        if t > self._latest:
            self._latest = t

            i = 0
            for ti, xi in self._events:
                if ti > t - self.period:
                    break
                self.dist.revert(xi)
                i += 1

            # Remove expired events
            if i > 0:
                self._events = self._events[i:]

        return self

    def revert(self, x, w=1):
        raise NotImplementedError

    @property
    def n_samples(self):
        return self.dist.n_samples

    def __repr__(self):
        return repr(self.dist)
