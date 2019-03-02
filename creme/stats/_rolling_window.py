import collections


class _RollingWindow:
    """Save k value to update windowed rolling statistics.

        Attributes:
            window_size (int): Size of the rolling window.
            rolling_window (deque): Store k current values.

    >>> from creme.stats._rolling_window import _RollingWindow

    >>> X = [1, 2, 3, 4, 5, 6]

    >>> _rolling_window = _RollingWindow(window_size=2)
    >>> for x in X:
    ...     print(_rolling_window.append(x)[0])
    1
    1
    2
    3
    4
    5

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.rolling_window = collections.deque([])

        if window_size < 2:
            raise ValueError(
                'window_size parameter must be stricly superior to 1.')

    def __getitem__(self, key):
        return self.rolling_window[key]

    def append(self, x):
        if len(self.rolling_window) + 1 <= self.window_size:
            self.rolling_window.append(x)
        else:
            self.rolling_window.popleft()
            self.rolling_window.append(x)
        return self
