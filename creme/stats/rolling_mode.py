import collections

from . import base
from . import _window


class RollingMode(base.RunningStatistic, _window.Window):
    """Running mode over a window.

    The mode is the most common value.

    Attributes:
        window_size (int): Size of the rolling window.
        counts (collections.defaultdict): Value counts.

    Example:

    ::

        >>> from creme import stats

        >>> X = ['sunny', 'sunny', 'sunny', 'rainy', 'rainy', 'rainy', 'rainy']
        >>> rolling_mode = stats.rolling_mode.RollingMode(window_size=2)
        >>> for x in X:
        ...     print(rolling_mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        rainy
        rainy
        rainy

        >>> rolling_mode = stats.rolling_mode.RollingMode(window_size=5)
        >>> for x in X:
        ...     print(rolling_mode.update(x).get())
        sunny
        sunny
        sunny
        sunny
        sunny
        rainy
        rainy

    """

    def __init__(self, window_size):
        super().__init__(window_size)
        self.counts = collections.defaultdict(int)

    @property
    def name(self):
        return f'rolling_{self.window_size}_mode'

    def update(self, x):
        if len(self.window) >= self.window_size:

            # Subtract the counter of the last element
            first_in = self.window[0]
            self.counts[first_in] -= 1

            # No need to store the value if it's counter is 0
            if self.counts[first_in] == 0:
                self.counts.pop(first_in)

        self.counts[x] += 1
        super().append(x)
        return self

    def get(self):
        return max(self.counts, key=self.counts.get)
