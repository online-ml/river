import bisect
import collections


class Window:
    """Maintains a list of previously seen values.

        Attributes:
            window_size (int)
            window (deque)

    >>> from creme.stats._window import Window

    >>> X = [1, 2, 3, 4, 5, 6]

    >>> window = Window(window_size=2)
    >>> for x in X:
    ...     print(window.append(x))
    [1]
    [1, 2]
    [2, 3]
    [3, 4]
    [4, 5]
    [5, 6]

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = collections.deque([], maxlen=window_size)

    def __getitem__(self, key):
        return self.window[key]

    def __len__(self):
        return len(self.window)

    def __repr__(self):
        return list(self.window).__repr__()

    def append(self, x):
        self.window.append(x)
        return self


class SortedWindow:
    """Compute the sorted running window.

    Attributes:
        window_size (int): size of the window to compute the rolling quantile.
        current_window (collections.deque): Store values that are in the current window.

    Example:

    ::

        >>> from creme.stats import _window
        >>> sorted_window = _window.SortedWindow(window_size=3)
        >>> for i in reversed(range(9)):
        ...     print(sorted_window.append(i))
        [8]
        [7, 8]
        [6, 7, 8]
        [5, 6, 7]
        [4, 5, 6]
        [3, 4, 5]
        [2, 3, 4]
        [1, 2, 3]
        [0, 1, 2]

    References:

    1. `Left sorted <https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python>`_

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = collections.deque([], maxlen=window_size)
        self.sorted_window = []

    def __getitem__(self, key):
        return self.sorted_window[key]

    def __len__(self):
        return len(self.window)

    def __repr__(self):
        return self.sorted_window.__repr__()

    def append(self, x):
        if len(self.sorted_window) >= self.window_size:
            self.sorted_window.remove(self.window[0])

        bisect.insort_left(self.sorted_window, x)
        self.window.append(x)

        return self
