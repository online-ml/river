from __future__ import annotations

from river import stats
from river.stats import _rust_stats


class Skew(stats.base.Univariate):
    """Running skew using Welford's algorithm.

    Parameters
    ----------
    bias
        If `False`, then the calculations are corrected for statistical bias.

    Examples
    --------

    >>> from river import stats
    >>> import numpy as np

    >>> np.random.seed(42)
    >>> X = np.random.normal(loc=0, scale=1, size=10)

    >>> skew = stats.Skew(bias=False)
    >>> for x in X:
    ...     print(skew.update(x).get())
    0.0
    0.0
    -1.4802398132849872
    0.5127437186677888
    0.7803466510704751
    1.056115628922055
    0.5057840774320389
    0.3478402420400934
    0.4536710660918704
    0.4123070197493227

    >>> skew = stats.Skew(bias=True)
    >>> for x in X:
    ...     print(skew.update(x).get())
    0.0
    0.0
    -0.6043053732501439
    0.2960327239981376
    0.5234724473423674
    0.7712778043924866
    0.39022088752624845
    0.278892645224261
    0.37425953513864063
    0.3476878073823696

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias
        self._skew = _rust_stats.RsSkew(bias)

    @property
    def name(self):
        return "skew"

    def update(self, x):
        self._skew.update(x)
        return self

    def get(self):
        return self._skew.get()
