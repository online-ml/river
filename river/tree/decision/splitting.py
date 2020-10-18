import abc
import collections
import functools
import operator

from river import base
from river import proba
from river import utils


def decimal_range(start, stop, num):
    """

    Examples
    --------

    >>> for x in decimal_range(0, 1, 4):
    ...     print(x)
    0.2
    0.4
    0.6
    0.8

    """
    step = (stop - start) / (num + 1)

    for _ in range(num):
        start += step
        yield start


class Op:

    __slots__ = 'symbol', 'func'

    def __init__(self, symbol, func):
        self.symbol = symbol
        self.func = func

    def __call__(self, a, b):
        return self.func(a, b)

    def __repr__(self):
        return self.symbol


LT = Op('<', operator.lt)
EQ = Op('=', operator.eq)


class SplitEnum(abc.ABC):

    @abc.abstractmethod
    def update(self, x, y):
        """Updates the sufficient statistics used for evaluating splits."""

    @abc.abstractmethod
    def enumerate_splits(self, target_dist):
        """Yields candidate split points and associated operators."""


class HistSplitEnum(SplitEnum):
    """Split enumerator for classification and numerical attributes."""

    def __init__(self, n_bins, n_splits):
        self.hists = collections.defaultdict(functools.partial(utils.Histogram, max_bins=n_bins))
        self.n_splits = n_splits

    def update(self, x: float, y: base.typing.ClfTarget):
        self.hists[y].update(x)
        return self

    def enumerate_splits(self, target_dist: proba.Multinomial):

        low = min(h[0].right for h in self.hists.values())
        high = min(h[-1].right for h in self.hists.values())

        # If only one single value has been observed, then no split can be proposed
        if low >= high:
            return
            yield  # not a typo

        n_thresholds = min(
            self.n_splits,
            max(map(len, self.hists.values())) - 1
        )

        thresholds = list(decimal_range(start=low, stop=high, num=n_thresholds))
        cdfs = {y: hist.iter_cdf(thresholds) for y, hist in self.hists.items()}

        for at in thresholds:

            l_dist = {}
            r_dist = {}

            for y in target_dist:
                p_xy = next(cdfs[y]) if y in cdfs else 0.  # P(x < t | y)
                p_y = target_dist.pmf(y)  # P(y)
                l_dist[y] = target_dist.n_samples * p_y * p_xy  # P(y | x < t)
                r_dist[y] = target_dist.n_samples * p_y * (1 - p_xy)  # P(y | x >= t)

            yield LT, at, proba.Multinomial(l_dist), proba.Multinomial(r_dist)


class CategoricalSplitEnum(SplitEnum):
    """Split enumerator for classification and categorical attributes."""

    def __init__(self):
        self.P_xy = collections.defaultdict(proba.Multinomial)

    def update(self, x: str, y: base.typing.ClfTarget):
        self.P_xy[y].update(x)
        return self

    def enumerate_splits(self, target_dist: proba.Multinomial):

        categories = set(*(p_x.keys() for p_x in self.P_xy.values()))

        # There has to be at least two categories for a split to be possible
        if len(categories) < 2:
            return
            yield  # not a typo

        for cat in categories:

            l_dist = {}
            r_dist = {}

            for y in target_dist:
                p_xy = self.P_xy[y].pmf(cat)  # P(cat | y)
                p_y = target_dist.pmf(y)  # P(y)
                l_dist[y] = target_dist.n_samples * p_y * p_xy  # P(y | cat)
                r_dist[y] = target_dist.n_samples * p_y * (1. - p_xy)  # P(y | !cat)

            yield EQ, cat, proba.Multinomial(l_dist), proba.Multinomial(r_dist)
