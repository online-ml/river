import abc
import collections
import functools
import math
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


class Split(collections.namedtuple('Split', 'on how at impurity')):
    """A data class for storing split details."""

    def __str__(self):
        return f'{self.on} {self.how} {self.at} ({self.impurity:.5f})'

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

    def enumerate_splits(self, target_dist, criterion):

        for b in self.P_x[:-1]:

            x = b.right
            p_x = self.P_x.cdf(x)  # P(x < t)

            l_dist = {}
            r_dist = {}

            for y in target_dist:
                p_xy = self.P_xy[y].cdf(x)  # P(x < t | y)
                p_y = target_dist.pmf(y)  # P(y)
                l_dist[y] = p_xy * p_y / p_x  # P(y | x < t)
                r_dist[y] = (1 - p_xy) * p_y / (1 - p_x)  # P(y | x >= t)

            impurity = p_x * criterion(l_dist) + (1 - p_x) * criterion(r_dist)
            yield Split(on=self.feature_name, how=LT, at=x, impurity=impurity)


class CategoricalSplitEnum(SplitEnum):
    """

    Example:

        >>> import collections
        >>> import random

        >>> sf = CategoricalSplitEnum()
        >>> class_counts = collections.Counter()

        >>> random.seed(42)
        >>> X = ['a'] * 30 + ['b'] * 30 + ['c'] * 20 + ['d'] * 20
        >>> Y = (
        ...     random.choices([0, 1, 2], weights=[0.3, 0.3, 0.4], k=30) +
        ...     random.choices([0, 1, 2], weights=[0.5, 0.2, 0.3], k=30) +
        ...     random.choices([0, 1, 2], weights=[0.1, 0.8, 0.1], k=20) +
        ...     random.choices([0, 1, 2], weights=[0.4, 0.2, 0.4], k=20)
        ... )

        >>> for x, y in zip(X, Y):
        ...     sf = sf.update(x, y)
        ...     class_counts.update([y])

    """

    def __init__(self, feature_name):
        super().__init__(feature_name)
        self.P_xy = collections.defaultdict(proba.Multinomial)
        self.categories = set()

    def update(self, x, y):
        self.P_xy[y].update(x)
        self.categories.add(x)
        return self

    def p_feature_given_class(self, x, y):
        return self.P_xy[y].pmf(x)

    def enumerate_splits(self):
        for cat in self.categories:
            yield cat, EQ


class GaussianSplitEnum(SplitEnum):
    """

    Example:

        >>> import collections
        >>> import random

        >>> sf = GaussianSplitEnum(n=10)
        >>> class_dist = collections.Counter()

        >>> random.seed(42)
        >>> X = (
        ...     [random.gauss(3, 2) for _ in range(60)] +
        ...     [random.gauss(7, 2) for _ in range(40)]
        ... )
        >>> Y = [False] * 60 + [True] * 40

        >>> for x, y in zip(X, Y):
        ...     sf = sf.update(x, y)
        ...     class_dist.update([y])

    """

    def __init__(self, feature_name, n):
        super().__init__(feature_name)
        self.P_xy = collections.defaultdict(proba.Gaussian)
        self.n = n
        self.min = math.inf
        self.max = -math.inf
        self.P_x = proba.Gaussian()

    def update(self, x, y):
        self.min = min(x, self.min)
        self.max = max(x, self.max)
        self.P_x.update(x)
        self.P_xy[y].update(x)
        return self

    def p_feature_given_class(self, x, y):
        return self.P_xy[y].cdf(x)

    def enumerate_splits(self):
        a = max(self.min, min(d.mu - 2. * d.sigma for d in self.P_xy.values()))
        b = min(self.max, max(d.mu + 2. * d.sigma for d in self.P_xy.values()))

        #a = max(self.min, self.P_x.mu - 2. * self.P_x.sigma)
        #b = min(self.max, self.P_x.mu + 2. * self.P_x.sigma)

        step = (b - a) / (self.n - 1)

        if step == 0:
            return

        for i in range(self.n):
            yield a + i * step, LT
