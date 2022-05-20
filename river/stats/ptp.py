from river import stats


class PeakToPeak(stats.base.Univariate):
    """Running peak to peak (max - min).

    Attributes
    ----------
    max : stats.Max
        The running max.
    min : stats.Min
        The running min.
    p2p : float
        The running peak to peak.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 4]
    >>> ptp = stats.PeakToPeak()
    >>> for x in X:
    ...     print(ptp.update(x).get())
    0
    5
    7
    7
    7
    8

    """

    def __init__(self):
        self.max = stats.Max()
        self.min = stats.Min()

    @property
    def name(self):
        return "ptp"

    def update(self, x):
        self.max.update(x)
        self.min.update(x)
        return self

    def get(self):
        return self.max.get() - self.min.get()


class RollingPeakToPeak(stats.base.RollingUnivariate):
    """Running peak to peak (max - min) over a window.

    Parameters
    ----------
    window_size
        Size of the rolling window.

    Attributes
    ----------
    max : stats.RollingMax
        The running rolling max.
    min : stats.RollingMin
        The running rolling min.

    Examples
    --------

    >>> from river import stats

    >>> X = [1, -4, 3, -2, 2, 1]
    >>> ptp = stats.RollingPeakToPeak(window_size=2)
    >>> for x in X:
    ...     print(ptp.update(x).get())
    0
    5
    7
    5
    4
    1

    """

    def __init__(self, window_size: int):
        self.max = stats.RollingMax(window_size)
        self.min = stats.RollingMin(window_size)

    @property
    def window_size(self):
        return self.max.window_size

    def update(self, x):
        self.max.update(x)
        self.min.update(x)
        return self

    def get(self):
        maximum = self.max.get()
        if maximum is None:
            return None
        minimum = self.min.get()
        if minimum is None:
            return None
        return maximum - minimum
