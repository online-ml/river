import abc
import collections
import functools
import operator

from .. import proba
from .. import utils


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

    def __init__(self, feature_name, n):
        super().__init__(feature_name)
        self.P_xy = collections.defaultdict(functools.partial(utils.Histogram, max_bins=n))
        self.P_x = utils.Histogram(max_bins=n)

    def update(self, x, y):
        self.P_xy[y].update(x)
        self.P_x.update(x)
        return self

    def enumerate_splits(self, target_dist):

        for b in self.P_x[:-1]:

            x = b.right

            l_dist = {}
            r_dist = {}

            for y in target_dist:
                p_xy = self.P_xy[y].cdf(x)  # P(x < t | y)
                p_y = target_dist.pmf(y)  # P(y)
                l_dist[y] = target_dist.n_samples * p_xy * p_y  # P(y | x < t)
                r_dist[y] = target_dist.n_samples * (1 - p_xy) * p_y  # P(y | x >= t)

            l_dist = proba.Multinomial(l_dist)
            r_dist = proba.Multinomial(r_dist)

            yield Split(on=self.feature_name, how=LT, at=x), l_dist, r_dist
