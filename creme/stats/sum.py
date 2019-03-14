from . import base


class Sum(base.RunningStatistic):
    """Running sum.

    Attributes:
        sum (float) : The running sum.

    """

    def __init__(self):
        self.sum = 0.

    @property
    def name(self):
        return 'sum'

    def update(self, x):
        self.sum += x
        return self

    def get(self):
        return self.sum
