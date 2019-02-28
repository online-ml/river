import bisect
import collections

from . import base


class _SortedWindow(base.RunningStatistic):
    """Compute the sorted running window.

    Attributes:
        window_size (int): size of the window to compute the rolling quantile.
        current_window (collections.deque): Store values that are in the current window.

    Example:

    ::

        >>> from creme.stats import _sorted_window
        >>> sorted_win = _sorted_window._SortedWindow(window_size=3)
        >>> for i in range(21):
        ...     sorted_win.update(i).get()
        [0]
        [0, 1]
        [0, 1, 2]
        [1, 2, 3]
        [2, 3, 4]
        [3, 4, 5]
        [4, 5, 6]
        [5, 6, 7]
        [6, 7, 8]
        [7, 8, 9]
        [8, 9, 10]
        [9, 10, 11]
        [10, 11, 12]
        [11, 12, 13]
        [12, 13, 14]
        [13, 14, 15]
        [14, 15, 16]
        [15, 16, 17]
        [16, 17, 18]
        [17, 18, 19]
        [18, 19, 20]

    References:

    1. `Left sorted <https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>`_

    """

    def __init__(self, window_size):

        self.window_size = window_size
        self.current_window = collections.deque([])
        self.sorted_window = []


    @property
    def name(self):
        return 'sorted_window'

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
        return self.sorted_window

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
