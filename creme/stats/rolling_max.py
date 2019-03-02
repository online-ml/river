from . import _sorted_window


class RollingMax(_sorted_window._SortedWindow):
    """Computes a windowed running max.

    Attributes:
        window_size (int): Size of the rolling window.

    Example:

        >>> from creme import stats

        >>> X = [1, -4, 3, -2, 2, 1]
        >>> rolling_max = stats.RollingMax(2)
        >>> for x in X:
        ...     print(rolling_max.update(x).get())
        1
        1
        3
        3
        2
        2

    """

    def __init__(self, window_size):
        super().__init__(window_size)

    @property
    def name(self):
        return 'rolling_max'

    def update(self, x):
        super().append(x)
        return self

    def get(self):
        return self.sorted_window[-1]
