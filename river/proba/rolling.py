import collections

from . import base


class Rolling(base.Distribution):
    """Wrapper for measuring probability distributions over a window.

    Parameters
    ----------
    dist
        A distribution.
    window_size
        A window size.

    Examples
    --------

    >>> import datetime as dt
    >>> from river import proba

    >>> X = ['red', 'green', 'green', 'blue', 'blue']

    >>> dist = proba.Rolling(
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

    """

    def __init__(self, dist: base.Distribution, window_size: int):
        self.dist = dist
        self.window_size = window_size
        self.window = collections.deque(maxlen=window_size)

    def update(self, x):
        if len(self.window) == self.window_size:
            self.dist.revert(self.window[0])
        self.dist.update(x)
        self.window.append(x)
        return self

    def revert(self, x, w=1):
        raise NotImplementedError

    @property
    def n_samples(self):
        return self.dist.n_samples

    def __repr__(self):
        return repr(self.dist)
