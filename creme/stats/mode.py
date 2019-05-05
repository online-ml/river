import collections

from . import base


__all__ = ['Mode']


class Mode(base.Univariate):
    """Running mode.

    The mode is simply the most common value. An approximate mode can be computed by setting the
    number of first unique values to count.

    Parameters:
        k (int): Only the first ``k`` unique values will be included. If ``k`` equals -1, the exact
            mode is computed.

    Attributes:
        counts (collections.defaultdict): Counts the number of occurrences of the different keys.

    Example:

        ::

            >>> from creme import stats

            >>> X = ['sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 'rainy']
            >>> mode = stats.Mode(k=2)
            >>> for x in X:
            ...     print(mode.update(x).get())
            sunny
            sunny
            cloudy
            cloudy
            cloudy
            cloudy

            >>> mode = stats.Mode(k=-1)
            >>> for x in X:
            ...     print(mode.update(x).get())
            sunny
            sunny
            cloudy
            cloudy
            cloudy
            rainy

    """

    def __init__(self, k=25):
        self.k = k
        self.counts = collections.defaultdict(int)

    @property
    def name(self):
        return 'mode'

    def update(self, x):
        if self.k == -1 or x in self.counts or len(self.counts) < self.k:
            self.counts[x] += 1
        return self

    def get(self):
        return max(self.counts, key=self.counts.get)
