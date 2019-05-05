from . import rolling_variance


class RollingSEM(rolling_variance.RollingVariance):
    """Running standard error of the mean over a window.

    Parameters:
        window_size (int): Size of the rolling window.
        ddof (int): Delta Degrees of Freedom for the variance.

    Example:

        ::
            >>> import creme

            >>> X = [1, 4, 2, -4, -8, 0]

            >>> rolling_sem = creme.stats.RollingSEM(ddof=1, window_size=2)
            >>> for x in X:
            ...     print(rolling_sem.update(x).get())
            0.0
            1.5
            1.0
            3.0
            2.0
            4.0

            >>> rolling_sem = creme.stats.RollingSEM(ddof=1, window_size=3)
            >>> for x in X:
            ...     print(rolling_sem.update(x).get())
            0.0
            1.5
            0.881917...
            2.403700...
            2.905932...
            2.309401...

    """

    @property
    def name(self):
        return f'rolling_{self.window_size}_sem'

    def get(self):
        return (super().get() / len(self.rolling_mean)) ** 0.5
