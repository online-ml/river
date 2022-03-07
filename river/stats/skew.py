from . import moments


class Skew(moments.CentralMoments):
    """Running skew using Welford's algorithm.

    Parameters
    ----------
    bias
        If `False`, then the calculations are corrected for statistical bias.

    Examples
    --------

    >>> import river.stats
    >>> import scipy.stats
    >>> import numpy as np

    >>> np.random.seed(42)
    >>> X = np.random.normal(loc=0, scale=1, size=10)

    >>> skew = river.stats.Skew(bias=False)
    >>> for x in X:
    ...     print(skew.update(x).get())
    0
    0.0
    -1.4802398132849872
    0.5127437186677888
    0.7803466510704751
    1.056115628922055
    0.5057840774320389
    0.3478402420400934
    0.4536710660918704
    0.4123070197493227

    >>> for i in range(1, len(X)+1):
    ...     print(scipy.stats.skew(X[:i], bias=False))
    0.0
    0.0
    -1.4802398132849874
    0.5127437186677893
    0.7803466510704746
    1.056115628922055
    0.5057840774320389
    0.3478402420400927
    0.4536710660918703
    0.4123070197493223

    >>> skew = river.stats.Skew(bias=True)
    >>> for x in X:
    ...     print(skew.update(x).get())
    0
    0.0
    -0.6043053732501439
    0.2960327239981376
    0.5234724473423674
    0.7712778043924866
    0.39022088752624845
    0.278892645224261
    0.37425953513864063
    0.3476878073823696

    >>> for i in range(1, len(X)+1):
    ...     print(scipy.stats.skew(X[:i], bias=True))
    0.0
    0.0
    -0.604305373250144
    0.29603272399813796
    0.5234724473423671
    0.7712778043924865
    0.39022088752624845
    0.2788926452242604
    0.3742595351386406
    0.34768780738236926

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)

    """

    def __init__(self, bias=False):
        super().__init__()
        self.bias = bias

    @property
    def name(self):
        return "skew"

    def update(self, x):
        self.count.update()
        self._update_delta(x)
        self._update_m1(x)
        self._update_sum_delta()
        self._update_m3()
        self._update_m2()
        return self

    def get(self):
        n = self.count.get()
        skew = n ** 0.5 * self.M3 / self.M2 ** 1.5 if self.M2 != 0 else 0
        if not self.bias and n > 2:
            return ((n - 1.0) * n) ** 0.5 / (n - 2.0) * skew
        return skew
