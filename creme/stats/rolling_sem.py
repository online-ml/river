from . import rolling_variance


class RollingSem(rolling_variance.RollingVariance):
    """Running standard error of the mean over a window.

    Parameters:
        window_size (int): Size of the rolling window.
        ddof (int): Factor used to correct the variance.

    >>> import creme

    >>> X = [1, 4, 2, -4, -8, 0]

    >>> rolling_sem = creme.stats.RollingSem(ddof=1, window_size=2)
    >>> for x in X:
    ...     print(rolling_sem.update(x).get())
    0.0
    1.499999...
    1.0
    2.999999...
    2.0
    4.0

    >>> rolling_sem = creme.stats.RollingSem(ddof=1, window_size=3)
    >>> for x in X:
    ...     print(rolling_sem.update(x).get())
    0.0
    1.499999...
    0.881917...
    2.403700...
    2.905932...
    2.309401...

    """

    def __init__(self, window_size, ddof=1):
        super().__init__(
            window_size=window_size,
            ddof=ddof,
        )

    @property
    def name(self):
        return f'rolling_{self.window_size}_sem'

    def get(self):
        return (super().get() ** 0.5) / (len(self.rolling_mean) ** 0.5)
