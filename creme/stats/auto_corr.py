import collections

from . import base
from . import pearson


class AutoCorrelation(base.Univariate, collections.deque):
    """Measures the serial correlation.

    This method computes the Pearson correlation between the current value and the value seen ``n``
    steps before.

    Example:

        The following examples are taken from the `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.autocorr.html>`_.

        ::

            >>> from creme import stats

            >>> auto_corr = stats.AutoCorrelation(lag=1)
            >>> for x in [0.25, 0.5, 0.2, -0.05]:
            ...     print(auto_corr.update(x).get())
            0
            0
            -1.0
            0.103552...

            >>> auto_corr = stats.AutoCorrelation(lag=2)
            >>> for x in [0.25, 0.5, 0.2, -0.05]:
            ...     print(auto_corr.update(x).get())
            0
            0
            0
            -1.0

            >>> auto_corr = stats.AutoCorrelation(lag=1)
            >>> for x in [1, 0, 0, 0]:
            ...     print(auto_corr.update(x).get())
            0
            0
            0
            0

    """

    def __init__(self, lag):
        super().__init__(maxlen=lag)
        self.lag = lag
        self.pearson = pearson.PearsonCorrelation(ddof=1)

    @property
    def name(self):
        return f'autocorr_{self.lag}'

    def update(self, x):

        # The correlation can be update once enough elements have been seen
        if len(self) == self.lag:
            self.pearson.update(x, self[0])

        # Add x to the window
        super().append(x)

        return self

    def get(self):
        return self.pearson.get()
