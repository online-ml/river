from . import base


class Count(base.Univariate):
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
