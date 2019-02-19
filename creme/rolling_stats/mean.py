from .. import stats
import collections

__all__ = ['RollingMean']


class RollingMean(stats.Mean):
    """Calculate the rolling average with a given window size.

    Attributes:
        window_size (int): size of the window to compute the rolling mean.
        current_window (collections.deque): Store values that are in the current window.

    >>> import creme

    >>> X = [1, 2, 3, 4, 5, 6]

    >>> rolling_mean = creme.rolling_stats.RollingMean(window_size=2)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5

    >>> rolling_mean = creme.rolling_stats.RollingMean(window_size=3)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.0
    3.0
    4.0
    5.0

    """

    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.current_window = collections.deque([])

    @property
    def name(self):
        return 'rolling_mean'

    def update(self, x):
        if len(self.current_window) + 1 > self.window_size:
            # Reset mean from out of window values.
            self.mean = self._update_mean(
                mean=self.mean,
                count_value=self.count.get(),
                current_window=self.current_window,
            )
            # Desincrement count from mean.
            self.count.desincrement()

        # Update current window.
        current_window = self._update_window(
            x=x,
            current_window=self.current_window,
            window_size=self.window_size,
        )
        # Update mean.
        super().update(x)
        return self

    def get(self):
        return self.mean

    @classmethod
    def _update_mean(cls, mean, count_value, current_window):
        mean = (mean * count_value -
                current_window[0]) / (count_value - 1)
        return mean

    @classmethod
    def _update_window(cls, x, current_window, window_size):
        if len(current_window) + 1 <= window_size:
            current_window.append(x)
        else:
            current_window.popleft()
            current_window.append(x)
        return current_window
