import bisect
import collections


class Window:
    """Maintains a list of previously seen values.

    Parameters:
        window_size (int):  Size of the rolling window.

    Attributes:
        window (collections.deque): Queue with a length `window_size`.

    Example:

        ::

            >>> from creme.stats.window import Window

            >>> X = [1, 2, 3, 4, 5, 6]

            >>> window = Window(window_size=2)
            >>> for x in X:
            ...     print(window.append(x))
            deque([1], maxlen=2)
            deque([1, 2], maxlen=2)
            deque([2, 3], maxlen=2)
            deque([3, 4], maxlen=2)
            deque([4, 5], maxlen=2)
            deque([5, 6], maxlen=2)

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.window = collections.deque([], maxlen=window_size)

    def __getitem__(self, key):
        return self.window[key]

    def __len__(self):
        return len(self.window)

    def __str__(self):
        return str(self.window)

    def append(self, x):
        self.window.append(x)
        return self


class SortedWindow:
    """Compute the sorted running window.

    Attributes:
        window_size (int): size of the window to compute the rolling quantile.
        currentwindow (collections.deque): Store values that are in the current window.

    Example:

    ::

        >>> from creme.stats import window
        >>> sortedwindow = window.SortedWindow(window_size=3)
        >>> for i in reversed(range(9)):
        ...     print(sortedwindow.append(i))
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
        self.sortedwindow = []

    def __getitem__(self, key):
        return self.sortedwindow[key]

    def __len__(self):
        return len(self.window)

    def __str__(self):
        return str(self.sortedwindow)

    def append(self, x):
        if len(self.sortedwindow) >= self.window_size:
            self.sortedwindow.remove(self.window[0])

        bisect.insort_left(self.sortedwindow, x)
        self.window.append(x)

        return self
