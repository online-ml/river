import abc
import collections
import functools
import operator

from .. import proba
from .. import utils


def decimal_range(start, stop, num):
    """

    Example:

        >>> for x in decimal_range(1, 3, 5):
        ...     print(x)
        1
        1.5
        2.0
        2.5
        3.0

    """
    step = (stop - start) / (num - 1)

    while start <= stop:
        yield start
        start += step


class Op(collections.namedtuple('Op', 'symbol operator')):

    def __call__(self, a, b):
        return self.operator(a, b)

    def __str__(self):
        return self.symbol


LT = Op('<', operator.lt)
EQ = Op('=', operator.eq)


class Split(collections.namedtuple('Split', 'on how at')):
    """A data class for storing split details."""

    def __str__(self):
        return f'{self.on} {self.how} {self.at}'

    def __call__(self, x):
        return self.how(x[self.on], self.at)


class SplitEnum(abc.ABC):

    def __init__(self, feature_name):
        self.feature_name = feature_name

    @abc.abstractmethod
    def update(self, x, y):
        """Updates the sufficient statistics used for evaluting splits."""

    @abc.abstractmethod
    def enumerate_splits(self):
        """Yields candidate split points and associated operators."""


class HistSplitEnum(SplitEnum, collections.defaultdict):

    def __init__(self, feature_name, n_bins, n_splits):
        super().__init__(feature_name)
        self.P_xy = collections.defaultdict(functools.partial(utils.Histogram, max_bins=n_bins))
        self.n_splits = n_splits

    def update(self, x, y):
        self.P_xy[y].update(x)
        return self

    def enumerate_splits(self, target_dist):

        a = min(h[0].right for h in self.P_xy.values())
        b = min(h[-1].right for h in self.P_xy.values())

        # If only one single value has been observed, then no split can be proposed
        if a >= b:
            return
            yield

        for x in decimal_range(start=a, stop=b, num=self.n_splits):

            l_dist = {}
            r_dist = {}

            for y in target_dist:
                p_xy = self.P_xy[y].cdf(x) if y in self.P_xy else 0.  # P(x < t | y)
                p_y = target_dist.pmf(y)  # P(y)
                l_dist[y] = target_dist.n_samples * p_xy * p_y  # P(y | x < t)
                r_dist[y] = target_dist.n_samples * (1 - p_xy) * p_y  # P(y | x >= t)

            l_dist = proba.Multinomial(l_dist)
            r_dist = proba.Multinomial(r_dist)

            yield Split(on=self.feature_name, how=LT, at=x), l_dist, r_dist
