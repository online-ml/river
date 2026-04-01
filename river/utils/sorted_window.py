from __future__ import annotations

import bisect
import collections
from typing import TypeVar

from river import base

T = TypeVar("T", bound=base.typing.SupportsComparison)


class SortedWindow(collections.UserList[T]):
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
    ...     window.append(i)
    ...     print(window)
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

    def __init__(self, size: int) -> None:
        super().__init__()
        self.unsorted_window: collections.deque[T] = collections.deque(maxlen=size)

    @property
    def size(self) -> int:
        return self.unsorted_window.maxlen  # type: ignore[return-value] # The window always has a maxlen

    def append(self, item: T) -> None:
        if len(self) >= self.size:
            # The window is sorted, and a binary search is more optimized than linear search
            start_deque = bisect.bisect_left(self, self.unsorted_window[0])
            del self[start_deque]

        bisect.insort_left(self, item)
        self.unsorted_window.append(item)
