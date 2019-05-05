from . import rolling_sum


class RollingMean(rolling_sum.RollingSum):
    """Running average over a window.

    Parameters:
        window_size (int): Size of the rolling window.

    Example:

        ::

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
        return f'rolling_{self.window_size}_mean'

    def get(self):
        return super().get() / len(self) if len(self) > 0 else 0
