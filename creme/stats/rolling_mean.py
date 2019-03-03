from . import rolling_sum


class RollingMean(rolling_sum.RollingSum):
    """Calculate the rolling average with a given window size.

    Attributes:
        window_size (int): Size of the rolling window.
        current_window (collections.deque): Store values that are in the current window.

    >>> import creme

    >>> X = [1, 2, 3, 4, 5, 6]

    >>> rolling_mean = creme.stats.RollingMean(window_size=2)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.5
    3.5
    4.5
    5.5

    >>> rolling_mean = creme.stats.RollingMean(window_size=3)
    >>> for x in X:
    ...     print(rolling_mean.update(x).get())
    1.0
    1.5
    2.0
    3.0
    4.0
    5.0

    """

    @property
    def name(self):
        return 'rolling_mean'

    def get(self):
        return super().get() / len(self)
