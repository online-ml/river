from __future__ import annotations

import math
import random

from river import base, stats

__all__ = ["KolmogorovSmirnov"]


class Treap(base.Base):
    """Class representing Treap (Cartesian Tree) used to calculate the Incremental KS statistics."""

    def __init__(self, key, value=0):
        self.key = key
        self.value = value
        self.priority = random.random()
        self.size = 1
        self.height = 1
        self.lazy = 0
        self.max_value = value
        self.min_value = value
        self.left = None
        self.right = None

    @staticmethod
    def sum_all(node, value):
        if node is None:
            return
        node.value += value
        node.max_value += value
        node.min_value += value
        node.lazy += value

    @classmethod
    def unlazy(cls, node):
        cls.sum_all(node.left, node.lazy)
        cls.sum_all(node.right, node.lazy)
        node.lazy = 0

    @classmethod
    def update(cls, node):
        if node is None:
            return
        cls.unlazy(node)
        node.size = 1
        node.height = 0
        node.max_value = node.value
        node.min_value = node.value

        if node.left is not None:
            node.size += node.left.size
            node.height += node.left.height
            node.max_value = max(node.max_value, node.left.max_value)
            node.min_value = min(node.min_value, node.left.min_value)

        if node.right is not None:
            node.size += node.right.size
            node.height = max(node.height, node.right.height)
            node.max_value = max(node.max_value, node.right.max_value)
            node.min_value = min(node.min_value, node.right.min_value)

        node.height += 1

    @classmethod
    def split_keep_right(cls, node, key):
        if node is None:
            return None, None

        left, right = None, None

        cls.unlazy(node)

        if key <= node.key:
            left, node.left = cls.split_keep_right(node.left, key)
            right = node
        else:
            node.right, right = cls.split_keep_right(node.right, key)
            left = node

        cls.update(left)
        cls.update(right)

        return left, right

    @classmethod
    def merge(cls, left, right):
        if left is None:
            return right
        if right is None:
            return left

        node = None

        if left.priority > right.priority:
            cls.unlazy(left)
            left.right = cls.merge(left.right, right)
            node = left
        else:
            cls.unlazy(right)
            right.left = cls.merge(left, right.left)
            node = right

        cls.update(node)

        return node

    @classmethod
    def split_smallest(cls, node):
        if node is None:
            return None, None

        left, right = None, None

        cls.unlazy(node)

        if node.left is not None:
            left, node.left = cls.split_smallest(node.left)
            right = node
        else:
            right = node.right
            node.right = None
            left = node

        cls.update(left)
        cls.update(right)

        return left, right

    @classmethod
    def split_greatest(cls, node):
        if node is None:
            return None, None

        cls.unlazy(node)

        if node.right is not None:
            node.right, right = cls.split_greatest(node.right)
            left = node
        else:
            left = node.left
            node.left = None
            right = node

        cls.update(left)
        cls.update(right)

        return left, right

    @staticmethod
    def get_size(node):
        return 0 if node is None else node.size

    @staticmethod
    def get_height(node):
        return 0 if node is None else node.height


class KolmogorovSmirnov(stats.base.Bivariate):
    r"""Incremental Kolmogorov-Smirnov statistics.

    The two-sample Kolmogorov-Smirnov test quantifies the distance between the empirical functions of two samples,
    with the null distribution of this statistic is calculated under the null hypothesis that the samples are drawn from
    the same distribution. The formula can be described as

    $$
    D_{n, m} = \sup_x \| F_{1, n}(x) - F_{2, m}(x) \|.
    $$

    This implementation is the incremental version of the previously mentioned statistics, with the change being in
    the ability to insert and remove an observation thorugh time. This can be done using a randomized tree called
    Treap (or Cartesian Tree) [^2] with bulk operation and lazy propagation.

    The implemented algorithm is able to perform the insertion and removal operations
    in O(logN) with high probability and calculate the Kolmogorov-Smirnov test in O(1),
    where N is the number of sample observations. This is a significant improvement compared
    to the O(N logN) cost of non-incremental implementation.

    This implementation also supports the calculation of the Kuiper statistics. Different from the orginial
    Kolmogorov-Smirnov statistics, Kuiper's test [^3] calculates the sum of the absolute sizes of the most positive and
    most negative differences between the two cumulative distribution functions taken into account. As such,
    Kuiper's test is very sensitive in the tails as at the median.

    Last but not least, this implementation is also based on the original implementation within the supplementary
    material of the authors of paper [^1], at
    [the following Github repository](https://github.com/denismr/incremental-ks/tree/master).

    Parameters
    ----------
    statistic
        The method used to calculate the statistic, can be either "ks" or "kuiper".
        The default value is set as "ks".

    Examples
    --------

    >>> import numpy as np
    >>> from river import stats

    >>> stream_a = [1, 1, 2, 2, 3, 3, 4, 4]
    >>> stream_b = [1, 1, 1, 1, 2, 2, 2, 2]

    >>> incremental_ks = stats.KolmogorovSmirnov(statistic="ks")
    >>> for a, b in zip(stream_a, stream_b):
    ...     incremental_ks.update(a, b)

    >>> incremental_ks
    KolmogorovSmirnov: 0.5

    >>> incremental_ks.n_samples
    8

    References
    ----------
    [^1]: dos Reis, D.M. et al. (2016) ‘Fast unsupervised online drift detection using incremental Kolmogorov-Smirnov
    test’, Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
    doi:10.1145/2939672.2939836.
    [^2]: C. R. Aragon and R. G. Seidel. Randomized search trees. In FOCS, pages 540–545. IEEE, 1989.
    [^3]: Kuiper, N. H. (1960). "Tests concerning random points on a circle".
    Proceedings of the Koninklijke Nederlandse Akademie van Wetenschappen, Series A. 63: 38–47.

    """

    def __init__(self, statistic="ks"):
        self.treap = None
        self.n_samples = 0
        self.statistic = statistic

    def update(self, x, y):
        keys = ((x, 0), (y, 1))

        self.n_samples += 1

        for key in keys:
            left, left_g, right, val = None, None, None, None

            left, right = Treap.split_keep_right(self.treap, key)
            left, left_g = Treap.split_greatest(left)
            val = 0 if left_g is None else left_g.value

            left = Treap.merge(left, left_g)
            right = Treap.merge(Treap(key, val), right)
            Treap.sum_all(right, 1 if key[1] == 0 else -1)

            self.treap = Treap.merge(left, right)

    def revert(self, x, y):
        keys = ((x, 0), (y, 1))

        self.n_samples -= 1

        for key in keys:
            left, right, right_l = None, None, None

            left, right = Treap.split_keep_right(self.treap, key)
            right_l, right = Treap.split_smallest(right)

            if right_l is not None and right_l.key == key:
                Treap.sum_all(right, -1 if key[1] == 0 else 1)
            else:
                right = Treap.merge(right_l, right)

            self.treap = Treap.merge(left, right)

    def get(self):
        assert self.statistic in ["ks", "kuiper"]
        if self.n_samples == 0:
            return 0

        if self.statistic == "ks":
            return max(self.treap.max_value, -self.treap.min_value) / self.n_samples
        elif self.statistic == "kuiper":
            return max(self.treap.max_value - self.treap.min_value) / self.n_samples
        else:
            raise ValueError(f"Unknown statistic {self.statistic}, expected one of: ks, kuiper")

    @staticmethod
    def _ca(p_value):
        return (-0.5 * math.log(p_value)) ** 0.5

    def _test_ks_threshold(self, ca):
        """
        Test whether the reference and sliding window follows the same or different probability distribution.
        This test will return `True` if we **reject** the null hypothesis that
        the two windows follow the same distribution.
        """
        return self.get() > ca * (2 * self.n_samples / self.n_samples**2) ** 0.5
