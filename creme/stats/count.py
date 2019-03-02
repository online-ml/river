from . import base


class Count(base.RunningStatistic):
    """Simply counts the number of times ``update`` is called."""

    def __init__(self):
        self.n = 0

    @property
    def name(self):
        return 'count'

    def update(self, x=None):
        self.n += 1
        return self

    def get(self):
        return self.n
