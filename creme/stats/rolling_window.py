import collections


class RollingWindow:
    """Save k value to update windowed rolling statistics.

        Attributes:
            window_size (int): Size of the rolling window.
            rolling_window (deque): Store k current values.

    >>> import creme

    >>> X = [1, 2, 3, 4, 5, 6]

    >>> rolling_window = creme.stats.RollingWindow(window_size=2)
    >>> for x in X:
    ...     print(rolling_window.update(x).get())
    deque([1])
    deque([1, 2])
    deque([2, 3])
    deque([3, 4])
    deque([4, 5])
    deque([5, 6])

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.rolling_window = collections.deque([])

        if window_size < 2:
            raise ValueError(
                'window_size parameter must be stricly superior to 1.')

    @property
    def name(self):
        return 'rolling_window'

    def get(self):
        return self.rolling_window

    def update(self, x):
        if len(self.rolling_window) + 1 <= self.window_size:
            self.rolling_window.append(x)
        else:
            self.rolling_window.popleft()
            self.rolling_window.append(x)
        return self
