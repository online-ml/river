import collections
import random

from .. import base
from .. import preprocessing


Limits = collections.namedtuple('Limits', 'lower upper')

Split = collections.namedtuple('Split', 'feature value')


class Node:

    def __init__(self, depth, size_limit, split=None, left=None, right=None):
        self.split = split
        self.size_limit = size_limit
        self.depth = depth
        self.left = left
        self.right = right
        self.r_mass = 0
        self.l_mass = 0

    @property
    def is_terminal(self):
        return self.left is None

    def next(self, x):
        if x[self.split.feature] < self.split.value:
            return self.left
        return self.right

    def update(self, x):

        # Increment the mass
        self.l_mass += 1

        # Update the next node
        if not self.is_terminal:
            self.next(x).update(x)

    def score(self, x):
        if self.r_mass < self.size_limit or self.is_terminal:
            return self.r_mass * 2 ** self.depth
        return self.next(x).score(x)

    def pivot_masses(self):
        self.r_mass = self.l_mass
        self.l_mass = 0

        if not self.is_terminal:
            self.left.pivot_masses()
            self.right.pivot_masses()

    def __str__(self):
        s = '\t' * self.depth + f'r_mass={self.r_mass}, l_mass={self.l_mass}'

        if self.is_terminal:
            return s

        return f'{s} ({self.split.feature} < {self.split.value:.3f})\n{self.left}\n{self.right}'


def make_limits(rng):
    sq = rng.random()
    return Limits(lower=sq - 2 * max(sq, 1 - sq), upper=sq + 2 * max(sq, 1 - sq))


def make_tree(limits, tree_height, depth, size_limit, rng):
    """Returns a half-space tree with splits occurring between 0 and 1.

    Parameters:
        limits (dict)
        max_depth (int): The desired tree height.
        depth (int)
        size_limit (int)

    """

    if depth == tree_height:
        return Node(depth=depth, size_limit=size_limit)

    # Randomly pick a feature and find the center of it's limits
    feature = rng.choice(list(limits.keys()))
    center = (limits[feature].lower + limits[feature].upper) / 2

    # Build the left node
    tmp = limits[feature]
    limits[feature] = Limits(lower=tmp.lower, upper=center)
    left = make_tree(
        limits=limits,
        tree_height=tree_height,
        depth=depth + 1,
        size_limit=size_limit,
        rng=rng
    )
    limits[feature] = tmp

    # Build the right node
    tmp = limits[feature]
    limits[feature] = Limits(lower=center, upper=tmp.upper)
    right = make_tree(
        limits=limits,
        tree_height=tree_height,
        depth=depth + 1,
        size_limit=size_limit,
        rng=rng
    )
    limits[feature] = tmp

    return Node(
        depth=depth,
        split=Split(feature=feature, value=center),
        left=left,
        right=right,
        size_limit=size_limit
    )


class HalfSpaceTrees(base.OutlierDetector):
    """Half-Space Trees (HST).

    Half-space trees are an online variant of isolation forests. They work well when anomalies are
    spread out in time. They do not work well if anomalies are packed together in windows of time,
    such as in the KDD'99 HTTP and SMTP datasets.

    Parameters:
        n_trees (int): Number of trees to use.
        tree_height (int): Height of each tree.
        window_size (int): Number of observations to use for calculating the mass at each node in
            each tree.
        scale (bool): Whether or not to scale features between 0 and 1. Only set to ``False`` if
            you know that all your features are already contained between 0 and 1.
        seed (int): Random number seed.

    Example:

        ::

            >>> from creme import anomaly

            >>> X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
            >>> hst = anomaly.HalfSpaceTrees(
            ...     n_trees=5,
            ...     tree_height=3,
            ...     window_size=3,
            ...     scale=False,
            ...     seed=42
            ... )

            >>> for x in X[:3]:
            ...     hst = hst.fit_one({'x': x})  # Warming up

            >>> for x in X:
            ...     features = {'x': x}
            ...     hst = hst.fit_one(features)
            ...     print(f'Anomaly score for {x:.3f}: {hst.score_one(features)}')
            Anomaly score for 0.500: 120
            Anomaly score for 0.450: 120
            Anomaly score for 0.430: 120
            Anomaly score for 0.440: 120
            Anomaly score for 0.445: 120
            Anomaly score for 0.450: 120
            Anomaly score for 0.000: 0

    References:
        1. `Fast Anomaly Detection for Streaming Data <https://www.ijcai.org/Proceedings/11/Papers/254.pdf>`_

    """

    def __init__(self, n_trees=25, tree_height=15, window_size=250, scale=True, seed=None):
        self.n_trees = n_trees
        self.window_size = window_size
        self.tree_height = tree_height
        self.scale = scale
        self.rng = random.Random(seed)
        self.trees = []
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.counter = 0

    def fit_one(self, x):

        # Scale the features between 0 and 1
        if self.scale:
            x = self.min_max_scaler.transform_one(x)
            self.min_max_scaler.fit_one(x)

        # The trees are built when the first observation comes in
        if not self.trees:
            self.trees = [
                make_tree(
                    limits={f: make_limits(rng=self.rng) for f in x},
                    tree_height=self.tree_height,
                    depth=0,
                    size_limit=.1 * self.window_size,
                    rng=self.rng
                )
                for _ in range(self.n_trees)
            ]

        # Update each tree
        for tree in self.trees:
            tree.update(x)

        # Pivot the masses if necessary
        self.counter += 1
        if self.counter == self.window_size:
            for tree in self.trees:
                tree.pivot_masses()
            self.counter = 0

        return self

    def score_one(self, x):
        if not self.trees:
            return 0
        return sum(tree.score(x) for tree in self.trees)
