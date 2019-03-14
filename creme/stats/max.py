import math

from . import base


class Max(base.RunningStatistic):
    """Running max.

    Attributes:
        max : The running max.

    """

    def __init__(self):
        self.max = -math.inf

    @property
    def name(self):
        return 'max'

    def update(self, x):
        if x > self.max:
            self.max = x
        return self

    def get(self):
        return self.max
