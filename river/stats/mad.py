from __future__ import annotations

from . import quantile


class MAD(quantile.Quantile):
    """Median Absolute Deviation (MAD).

    The median absolute deviation is the median of the absolute differences between each data point
    and the data's overall median. In an online setting, the median of the data is unknown
    beforehand. Therefore, both the median of the data and the median of the differences of the
    data with respect to the latter are updated online. To be precise, the median of the data is
    updated before the median of the differences. As a consequence, this online version of the MAD
    does not coincide exactly with its batch counterpart.

    Examples
    --------

    >>> from river import stats

    >>> X = [4, 2, 5, 3, 0, 4]

    >>> mad = stats.MAD()
    >>> for x in X:
    ...     print(mad.update(x).get())
    0.0
    2.0
    1.0
    1.0
    1.0
    1.0

    Attributes
    ----------
    median : stats.Median
        The median of the data.

    References
    ----------
    [^1]: [Median absolute deviation article on Wikipedia](https://www.wikiwand.com/en/Median_absolute_deviation)

    """

    #
    def __init__(self):
        super().__init__(q=0.5)
        self.median = quantile.Quantile(q=0.5)

    def update(self, x):
        self.median.update(x)
        super().update(abs(x - self.median.get()))
        return self
