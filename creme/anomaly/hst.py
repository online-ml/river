import collections
import functools
import operator
import random
import typing

from creme import base
from creme import preprocessing
from creme.tree.base import Leaf, Branch, Split


__all__ = ['HalfSpaceTrees']


def make_padded_tree(limits, height, padding, rng=random, **node_params):

    if height == 0:
        return Leaf(**node_params)

    # Randomly pick a feature
    # We weight each feature by the gap between each feature's limits
    on = rng.choices(
        population=list(limits.keys()),
        weights=[limits[i][1] - limits[i][0] for i in limits]
    )[0]

    # Pick a split point; use padding to avoid too narrow a split
    a = limits[on][0]
    b = limits[on][1]
    at = rng.uniform(a + padding * (b - a), b - padding * (b - a))

    # Build the left node
    tmp = limits[on]
    limits[on] = (tmp[0], at)
    left = make_padded_tree(limits=limits, height=height - 1, padding=padding, rng=rng, **node_params)
    limits[on] = tmp

    # Build the right node
    tmp = limits[on]
    limits[on] = (at, tmp[1])
    right = make_padded_tree(limits=limits, height=height - 1, padding=padding, rng=rng, **node_params)
    limits[on] = tmp

    split = Split(on=on, how=operator.lt, at=at)
    return Branch(split=split, left=left, right=right, **node_params)


class HalfSpaceTrees(base.AnomalyDetector):
    """Half-Space Trees (HST).

    Half-space trees are an online variant of isolation forests. They work well when anomalies are
    spread out in time. They do not work well if anomalies are packed together in windows of time,
    such as in the KDD'99 HTTP and SMTP datasets.

    Parameters:
        n_trees: Number of trees to use.
        height: Height of each tree. Note that a tree of height `h` is made up of `h + 1`
            levels and therefore contains `2 ** (h + 1) - 1` nodes.
        window_size: Number of observations to use for calculating the mass at each node in
            each tree.
        limits: Specifies the range of each feature. By default each feature is assumed to
            be in range `[0, 1]`.
        seed: Random number seed.

    Example:

        >>> from creme import anomaly

        >>> X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
        >>> hst = anomaly.HalfSpaceTrees(
        ...     n_trees=5,
        ...     height=3,
        ...     window_size=3,
        ...     seed=42
        ... )

        >>> for x in X[:3]:
        ...     hst = hst.fit_one({'x': x})  # Warming up

        >>> for x in X:
        ...     features = {'x': x}
        ...     hst = hst.fit_one(features)
        ...     print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')
        Anomaly score for x=0.500: 0.107
        Anomaly score for x=0.450: 0.071
        Anomaly score for x=0.430: 0.107
        Anomaly score for x=0.440: 0.107
        Anomaly score for x=0.445: 0.107
        Anomaly score for x=0.450: 0.071
        Anomaly score for x=0.000: 0.853

    References:
        1. [Tan, S.C., Ting, K.M. and Liu, T.F., 2011, June. Fast anomaly detection for streaming data. In Twenty-Second International Joint Conference on Artificial Intelligence.](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

    """

    def __init__(self, n_trees=25, height=15, window_size=250,
                 limits: typing.Dict[base.typing.FeatureName, typing.Tuple[float, float]] = None,
                 seed: int = None):

        self.n_trees = n_trees
        self.window_size = window_size
        self.height = height
        self.limits = collections.defaultdict(functools.partial(tuple, (0, 1)))
        if limits is not None:
            self.limits.update(limits)
        self.seed = seed
        self.rng = random.Random(seed)

        self.trees = []
        self.counter = 0
        self._first_window = True

    @property
    def size_limit(self):
        """This is the threshold under which the node search stops during the scoring phase.

        The value .1 is a magic constant indicated in the original paper.

        """
        return .1 * self.window_size

    @property
    def _max_score(self):
        """The largest potential anomaly score."""
        return self.n_trees * self.window_size * (2 ** (self.height + 1) - 1)

    def fit_one(self, x):

        # The trees are built when the first observation comes in
        if not self.trees:
            self.trees = [
                make_padded_tree(
                    limits={i: self.limits[i] for i in x},
                    height=self.height,
                    padding=.15,
                    rng=self.rng,
                    # kwargs
                    r_mass=0,
                    l_mass=0
                )
                for _ in range(self.n_trees)
            ]

        # Update each tree
        for tree in self.trees:
            for node in tree.path(x):
                node.l_mass += 1

        # Pivot the masses if necessary
        self.counter += 1
        if self.counter == self.window_size:
            for tree in self.trees:
                for node, _ in tree.iter_dfs():
                    node.r_mass = node.l_mass
                    node.l_mass = 0
            self._first_window = False
            self.counter = 0

        return self

    def score_one(self, x):

        if self._first_window:
            return 0

        score = 0.
        for tree in self.trees:
            for depth, node in enumerate(tree.path(x)):
                score += node.r_mass * 2 ** depth
                if node.r_mass < self.size_limit:
                    break

        # Normalize the score between 0 and 1
        score /= self._max_score

        # We want high score -> anomaly, but we have high score -> normal
        return 1 - score
