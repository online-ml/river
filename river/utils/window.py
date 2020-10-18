import bisect
import collections
import typing


class Window:
    """Running window data structure.

    This is just a convenience layer on top of a `collections.deque`. The only reason this exists
    is that deepcopying a class which inherits from `collections.deque` seems to bug out when the
    class has a parameter with no default value.

    Parameters
    ----------
    size
        Size of the rolling window.

    Examples
    --------

    >>> from river import utils

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

    def __init__(self, size: int):
        self.values: typing.Deque[typing.Any] = collections.deque(maxlen=size)

    @property
    def size(self):
        return self.values.maxlen

    def __repr__(self):
        return str(list(self.values))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = val
        return self

    def extend(self, values):
        self.values.extend(values)
        return self

    def append(self, x):
        self.values.append(x)
        return self

    def popleft(self):
        return self.values.popleft()


class SortedWindow(collections.UserList):
    """Sorted running window data structure.

    Parameters
    ----------
    size
        Size of the window to compute the rolling quantile.

    Examples
    --------

    >>> from river import utils

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

    References
    ----------
    [^1]: [Left sorted inserts in Python](https://stackoverflow.com/questions/8024571/insert-an-item-into-sorted-list-in-python)

    """

    def __init__(self, size: int):
        super().__init__()
        self.unsorted_window = Window(size)

    @property
    def size(self):
        return self.unsorted_window.size

    def append(self, x):

        if len(self) >= self.size:
            self.remove(self.unsorted_window[0])

        bisect.insort_left(self, x)
        self.unsorted_window.append(x)

        return self
