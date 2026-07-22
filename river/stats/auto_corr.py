from __future__ import annotations

import collections

from river import stats


class AutoCorr(stats.base.Univariate):
    """Measures the serial correlation.

    This method computes the Pearson correlation between the current value and the value seen `n`
    steps before.

    Parameters
    ----------
    lag

    Examples
    --------

    The following examples are taken from the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.autocorr.html).

    >>> from river import stats

    >>> auto_corr = stats.AutoCorr(lag=1)
    >>> for x in [0.25, 0.5, 0.2, -0.05]:
    ...     auto_corr.update(x)
    ...     print(auto_corr.get())
    0
    0
    -1.0
    0.103552

    >>> auto_corr = stats.AutoCorr(lag=2)
    >>> for x in [0.25, 0.5, 0.2, -0.05]:
    ...     auto_corr.update(x)
    ...     print(auto_corr.get())
    0
    0
    0
    -1.0

    >>> auto_corr = stats.AutoCorr(lag=1)
    >>> for x in [1, 0, 0, 0]:
    ...     auto_corr.update(x)
    ...     print(auto_corr.get())
    0
    0
    0
    0

    """

    def __init__(self, lag: int) -> None:
        self.window: collections.deque[float] = collections.deque(maxlen=lag)
        self.lag: int = lag
        self.pearson: stats.PearsonCorr = stats.PearsonCorr(ddof=1)

    @property
    def name(self) -> str:
        return f"autocorr_{self.lag}"

    def update(self, x: float) -> None:
        # The correlation can be update once enough elements have been seen
        if len(self.window) == self.lag:
            self.pearson.update(x, self.window[0])

        # Add x to the window
        self.window.append(x)

    def get(self) -> float:
        return self.pearson.get()
