from . import base
from . import count


class Mean(base.RunningStatistic):
    """Computes a running mean.

    Attributes:
        count (stats.Count)
        mu (float): The current estimated mean.

    """

    def __init__(self):
        self.count = count.Count()
        self.mu = 0

    @property
    def name(self):
        return 'mean'

    def update(self, x):
        self.mu += (x - self.mu) / self.count.update().get()
        return self

    def get(self):
        return self.mu
