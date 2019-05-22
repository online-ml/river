import bisect
import collections


class Window(collections.deque):
    """Running window data structure.

    This is just a convenience layer on top of a `collections.deque`.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

            >>> from creme import utils

            >>> window = utils.Window(size=2)

            >>> for x in [1, 2, 3, 4, 5, 6]:
            ...     print(window.append(x))
            [1]
            [1, 2]
            [2, 3]
            [3, 4]
            [4, 5]
            [5, 6]

    """

    def __init__(self, size):
        super().__init__([], maxlen=size)

    def __repr__(self):
        return str(list(self))

    @property
    def size(self):
        return self.maxlen

    def append(self, x):
        super().append(x)
        return self


class SortedWindow(collections.UserList):
    """Sorted running window data structure.

    Parameters:
        window_size (int): size of the window to compute the rolling quantile.

    Example:

        ::

            >>> from creme import utils

            >>> window = utils.SortedWindow(size=3)

            >>> for i in reversed(range(9)):
            ...     print(window.append(i))
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

    def __init__(self, size):
        super().__init__()
        self.unsorted_window = Window(size)

    @property
    def size(self):
        return self.unsorted_window.size

    def append(self, x):

        if len(self) >= self.unsorted_window.maxlen:
            self.remove(self.unsorted_window[0])

        bisect.insort_left(self, x)
        self.unsorted_window.append(x)

        return self
