from . import base
from . import max
from . import min


class PeakToPeak(base.Univariate):
    """Running peak to peak (max - min).

    Attributes:
        max (stats.Max): The running max.
        min (stats.Min): The running min.
        p2p (float): The running peak to peak.

    Example:

        ::

            >>> from creme import stats

            >>> X = [1, -4, 3, -2, 2, 4]
            >>> ptp = stats.PeakToPeak()
            >>> for x in X:
            ...     print(ptp.update(x).get())
            0
            5
            7
            7
            7
            8

    """

    def __init__(self):
        self.max = max.Max()
        self.min = min.Min()

    @property
    def name(self):
        return 'ptp'

    def update(self, x):
        self.max.update(x)
        self.min.update(x)
        return self

    def get(self):
        return self.max.get() - self.min.get()
