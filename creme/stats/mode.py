import collections

from . import base


class Mode(base.RunningStatistic):
    """Running mode.

    Parameters:
        k (int): Only the first ``k`` unique values will be included.
        exact (bool): Whether or not to count the occurrences of each and every value.

    Attributes:
        counts (collections.defaultdict)

    Example:

    ::

        >>> from creme import stats

        >>> X = ['sunny', 'cloudy', 'cloudy', 'rainy', 'rainy', 'rainy']
        >>> mode = stats.mode.Mode(k=2, exact=False)
        >>> for x in X:
        ...     print(mode.update(x).get())
        sunny
        sunny
        cloudy
        cloudy
        cloudy
        cloudy

        >>> mode = stats.mode.Mode(k=2, exact=True)
        >>> for x in X:
        ...     print(mode.update(x).get())
        sunny
        sunny
        cloudy
        cloudy
        cloudy
        rainy

    """

    def __init__(self, k=25, exact=False):
        self.k = k
        self.counts = collections.defaultdict(int)
        self.exact = exact

    @property
    def name(self):
        return 'mode'

    def update(self, x):
        if x in self.counts or len(self.counts) < self.k or self.exact:
            self.counts[x] += 1
        return self

    def get(self):
        return max(self.counts, key=self.counts.get)
