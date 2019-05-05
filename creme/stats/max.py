import math

from . import base


class Max(base.Univariate):
    """Running max.

    Attributes:
        max (float): The running max.

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
