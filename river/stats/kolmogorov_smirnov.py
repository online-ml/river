from __future__ import annotations

import math
import random

from river import stats

__all__ = ["KolmogorovSmirnov"]

_random = random.random


class _Node:
    """Treap node with __slots__ for fast attribute access."""

    __slots__ = (
        "key",
        "value",
        "priority",
        "size",
        "height",
        "lazy",
        "max_value",
        "min_value",
        "left",
        "right",
    )

    def __init__(self, key, value=0):
        self.key = key
        self.value = value
        self.priority = _random()
        self.size = 1
        self.height = 1
        self.lazy = 0
        self.max_value = value
        self.min_value = value
        self.left = None
        self.right = None


def _push_down(node):
    """Push lazy value to children (inline unlazy + sum_all)."""
    lz = node.lazy
    if lz != 0:
        left = node.left
        if left is not None:
            left.value += lz
            left.max_value += lz
            left.min_value += lz
            left.lazy += lz
        right = node.right
        if right is not None:
            right.value += lz
            right.max_value += lz
            right.min_value += lz
            right.lazy += lz
        node.lazy = 0


def _pull_up(node):
    """Recompute aggregate fields from children."""
    if node is None:
        return
    _push_down(node)
    val = node.value
    mx = val
    mn = val
    sz = 1
    ht = 0
    left = node.left
    if left is not None:
        l_mx = left.max_value
        l_mn = left.min_value
        sz += left.size
        ht = left.height
        if l_mx > mx:
            mx = l_mx
        if l_mn < mn:
            mn = l_mn
    right = node.right
    if right is not None:
        r_mx = right.max_value
        r_mn = right.min_value
        sz += right.size
        r_ht = right.height
        if r_ht > ht:
            ht = r_ht
        if r_mx > mx:
            mx = r_mx
        if r_mn < mn:
            mn = r_mn
    node.size = sz
    node.height = ht + 1
    node.max_value = mx
    node.min_value = mn


def _split_keep_right(node, key):
    if node is None:
        return None, None
    _push_down(node)
    if key <= node.key:
        left, node.left = _split_keep_right(node.left, key)
        right = node
    else:
        node.right, right = _split_keep_right(node.right, key)
        left = node
    _pull_up(left)
    _pull_up(right)
    return left, right


def _merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        _push_down(left)
        left.right = _merge(left.right, right)
        _pull_up(left)
        return left
    else:
        _push_down(right)
        right.left = _merge(left, right.left)
        _pull_up(right)
        return right


def _split_smallest(node):
    if node is None:
        return None, None
    _push_down(node)
    if node.left is not None:
        left, node.left = _split_smallest(node.left)
        right = node
    else:
        right = node.right
        node.right = None
        left = node
    _pull_up(left)
    _pull_up(right)
    return left, right


def _split_greatest(node):
    if node is None:
        return None, None
    _push_down(node)
    if node.right is not None:
        node.right, right = _split_greatest(node.right)
        left = node
    else:
        left = node.left
        node.left = None
        right = node
    _pull_up(left)
    _pull_up(right)
    return left, right


def _sum_all(node, value):
    if node is not None:
        node.value += value
        node.max_value += value
        node.min_value += value
        node.lazy += value


class KolmogorovSmirnov(stats.base.Bivariate):
    r"""Incremental Kolmogorov-Smirnov statistics.

    The two-sample Kolmogorov-Smirnov test quantifies the distance between the empirical functions of two samples,
    with the null distribution of this statistic is calculated under the null hypothesis that the samples are drawn from
    the same distribution. The formula can be described as

    $$
    D_{n, m} = \sup_x \| F_{1, n}(x) - F_{2, m}(x) \|.
    $$

    This implementation is the incremental version of the previously mentioned statistics, with the change being in
    the ability to insert and remove an observation through time. This can be done using a randomized tree called
    Treap (or Cartesian Tree) [^2] with bulk operation and lazy propagation.

    The implemented algorithm is able to perform the insertion and removal operations
    in O(logN) with high probability and calculate the Kolmogorov-Smirnov test in O(1),
    where N is the number of sample observations. This is a significant improvement compared
    to the O(N logN) cost of non-incremental implementation.

    This implementation also supports the calculation of the Kuiper statistics. Different from the original
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
    [^1]: dos Reis, D.M. et al. (2016) 'Fast unsupervised online drift detection using incremental Kolmogorov-Smirnov
    test', Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
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
        root = self.treap
        self.n_samples += 1

        for key, delta in (((x, 0), 1), ((y, 1), -1)):
            left, right = _split_keep_right(root, key)
            left, left_g = _split_greatest(left)
            val = 0 if left_g is None else left_g.value

            left = _merge(left, left_g)
            new_node = _Node(key, val)
            right = _merge(new_node, right)
            _sum_all(right, delta)

            root = _merge(left, right)

        self.treap = root

    def revert(self, x, y):
        root = self.treap
        self.n_samples -= 1

        for key, delta in (((x, 0), -1), ((y, 1), 1)):
            left, right = _split_keep_right(root, key)
            right_l, right = _split_smallest(right)

            if right_l is not None and right_l.key == key:
                _sum_all(right, delta)
            else:
                right = _merge(right_l, right)

            root = _merge(left, right)

        self.treap = root

    def get(self):
        assert self.statistic in ["ks", "kuiper"]
        if self.n_samples == 0:
            return 0

        if self.statistic == "ks":
            return max(self.treap.max_value, -self.treap.min_value) / self.n_samples
        elif self.statistic == "kuiper":
            return (self.treap.max_value - self.treap.min_value) / self.n_samples
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
