import math

from . import base


class Min(base.Univariate):
    """Running min.

    Attributes:
        min (float): The running min.

    """

    def __init__(self):
        self.min = math.inf

    @property
    def name(self):
        return 'min'

    def update(self, x):
        if x < self.min:
            self.min = x
        return self

    def get(self):
        return self.min
