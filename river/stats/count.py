from __future__ import annotations

from river import stats


class Count(stats.base.Univariate):
    """A simple counter.

    Attributes
    ----------
    n : int
        The current number of observations.

    """

    def __init__(self):
        self.n = 0

    def update(self, x=None):
        self.n += 1
        return self

    def get(self):
        return self.n
