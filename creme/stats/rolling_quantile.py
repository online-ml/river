import collections
import bisect

from . import base



class RollingQuantile(base.RunningStatistic):
    """Calculate the rolling quantile with a given window size.

    Attributes:
        percentile (float): Percentile you want compute the value 
            must be between 0 and 1 excluded.
        window_size (int): size of the window to compute the rolling quantile.
        current_window (collections.deque): Store values that are in the current window.

    Example:

        ::
            >>> from creme import stats
            >>> import numpy as np

            >>> rolling_quantile = stats.RollingQuantile(window_size = 100,
            ...                                          percentile = 0.5)

            >>> for i in range(0,1001):
            ...     _ = rolling_quantile.update(i)
            ...     if i%100 == 0:
            ...         print(rolling_quantile.get()) 
            0
            50
            150
            250
            350
            450
            550
            650
            750
            850
            950



    References:

    - `Left sorted <https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>`_


    """

    def __init__(self, window_size, percentile=0.5):

        if 0 < percentile < 1:
            self.percentile = percentile
        else:
            raise ValueError('percentile must be between 0 and 1 excluded')

        self.percentile = percentile
        self.window_size = window_size
        self.current_window = collections.deque([])
        self.sorted_window = []

        self.idx_percentile = int(round(self.percentile * self.window_size + 0.5)) - 1

    @property
    def name(self):
        return 'rolling_quantile'

    def update(self, x):
        # Update current window.
        self.current_window, self.sorted_window = self._update_window(
            x=x,
            current_window=self.current_window,
            sorted_window=self.sorted_window,
            window_size=self.window_size,
        )

        return self

    def get(self):
        if len(self.current_window) < self.window_size:
            _idx_percentile = int(round(self.percentile * len(self.current_window) + 0.5)) - 1
            return self.sorted_window[_idx_percentile]

        return self.sorted_window[self.idx_percentile]

    @classmethod
    def _update_window(cls, x, current_window, sorted_window, window_size):
        if len(current_window) < window_size:
            current_window.append(x)
            bisect.insort_left(sorted_window, x)

        else:
            remove_sorted = current_window[0]
            sorted_window.remove(remove_sorted)
            bisect.insort_left(sorted_window, x)
            current_window.popleft()
            current_window.append(x)
        return current_window, sorted_window
