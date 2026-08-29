from __future__ import annotations

import random

from river.tree.base import Branch, Leaf

__all__ = ["PaddedBranch", "PaddedLeaf", "make_padded_tree"]


class PaddedBranch(Branch):
    def __init__(self, left, right, feature, threshold, l_mass, r_mass):
        super().__init__(left, right)
        self.feature = feature
        self.threshold = threshold
        self.l_mass = l_mass
        self.r_mass = r_mass

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    def next(self, x):
        """

        We want to handle the case where a split feature is missing. In that case, we go down the
        child that has been the most visited in the past.

        """
        left, right = self.children
        try:
            value = x[self.feature]
        except KeyError:
            if left.l_mass < right.l_mass:
                return right
            return left
        if value < self.threshold:
            return left
        return right

    def most_common_path(self):
        raise NotImplementedError

    @property
    def repr_split(self):
        return f"{self.feature} < {self.threshold:.5f}"


class PaddedLeaf(Leaf):
    def __repr__(self):
        return str(self.r_mass)


def make_padded_tree(limits, height, padding, rng=random, **node_params):
    if height == 0:
        return PaddedLeaf(**node_params)

    # Randomly pick a feature
    # We weight each feature by the gap between each feature's limits
    on = rng.choices(
        population=list(limits.keys()),
        weights=[limits[i][1] - limits[i][0] for i in limits],
    )[0]

    # Pick a split point; use padding to avoid too narrow a split
    a = limits[on][0]
    b = limits[on][1]
    at = rng.uniform(a + padding * (b - a), b - padding * (b - a))

    # Build the left node
    tmp = limits[on]
    limits[on] = (tmp[0], at)
    left = make_padded_tree(
        limits=limits, height=height - 1, padding=padding, rng=rng, **node_params
    )
    limits[on] = tmp

    # Build the right node
    tmp = limits[on]
    limits[on] = (at, tmp[1])
    right = make_padded_tree(
        limits=limits, height=height - 1, padding=padding, rng=rng, **node_params
    )
    limits[on] = tmp

    return PaddedBranch(left=left, right=right, feature=on, threshold=at, **node_params)
