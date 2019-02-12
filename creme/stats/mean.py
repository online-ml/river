from . import base
from . import count


class Mean(base.RunningStatistic):
    """Computes a running mean.

    Attributes:
        count (stats.Count)
        mean (float)

    Example:

        >>> from creme import stats

        >>> X = [-5, -3, -1, 1, 3, 5]
        >>> mean = stats.Mean()
        >>> for x in X:
        ...     print(mean.update(x).get())
        -5.0
        -4.0
        -3.0
        -2.0
        -1.0
        0.0

    Attributes:
        count (stats.Count)
        mean (float): The current estimated mean.

    """

    def __init__(self):
        self.count = count.Count()
        self.mean = None

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        n = self.count.update(x).get()
        self.mean = float(
            x) if self.mean is None else self.mean + (x - self.mean) / n
        return self

    def get(self):
        return self.mean
