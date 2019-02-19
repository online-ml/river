from . import base
from . import max
from . import min


class PeakToPeak(base.RunningStatistic):
    """Computes a running peak to peak (max - min).

    Attributes:
        max (stats.Max)
        min (stats.Min)
        p2p (float): The running peak to peak.

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
