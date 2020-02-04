import operator
import random

from .. import base
from .. import preprocessing
from ..tree.base import Leaf, Branch, Split


__all__ = ['HalfSpaceTrees']


def make_tree(limits, height, rng=random, **node_params):

    if height == 0:
        return Leaf(**node_params)

    # Randomly pick a feature and find the center of it's limits
    feature = rng.choice(list(limits.keys()))
    center = (limits[feature][0] + limits[feature][1]) / 2

    # Build the left node
    tmp = limits[feature]
    limits[feature] = (tmp[0], center)
    left = make_tree(limits=limits, height=height - 1, rng=rng, **node_params)
    limits[feature] = tmp

    # Build the right node
    tmp = limits[feature]
    limits[feature] = (center, tmp[1])
    right = make_tree(limits=limits, height=height - 1, rng=rng, **node_params)
    limits[feature] = tmp

    split = Split(on=feature, how=operator.lt, at=center)
    return Branch(split=split, left=left, right=right, **node_params)


def make_limits(rng):
    sq = rng.random()
    return (
        sq - 2 * max(sq, 1 - sq),
        sq + 2 * max(sq, 1 - sq)
    )


class HalfSpaceTrees(base.AnomalyDetector):
    """Half-Space Trees (HST).

    Half-space trees are an online variant of isolation forests. They work well when anomalies are
    spread out in time. They do not work well if anomalies are packed together in windows of time,
    such as in the KDD'99 HTTP and SMTP datasets.

    Parameters:
        n_trees (int): Number of trees to use.
        height (int): Height of each tree.
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
            ...     height=3,
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
            Anomaly score for 0.500: 1.0
            Anomaly score for 0.450: 1.0
            Anomaly score for 0.430: 1.0
            Anomaly score for 0.440: 1.0
            Anomaly score for 0.445: 1.0
            Anomaly score for 0.450: 1.0
            Anomaly score for 0.000: 0.0

    References:
        1. `Fast Anomaly Detection for Streaming Data <https://www.ijcai.org/Proceedings/11/Papers/254.pdf>`_

    """

    def __init__(self, n_trees=25, height=15, window_size=250, scale=True, seed=None):
        self.n_trees = n_trees
        self.window_size = window_size
        self.height = height
        self.scale = scale
        self.seed = seed
        self.rng = random.Random(seed)
        self.trees = []
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.counter = 0

    @property
    def size_limit(self):
        return .1 * self.window_size

    @property
    def _max_score(self):
        """The maximum possible score.

        We obtain this by looking at the extreme case where all the samples from a particular
        window have fallen the same leaf within each tree.

        """
        return self.n_trees * self.window_size * 2 ** self.height

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
                    height=self.height,
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
                for node in tree.path(x):
                    node.r_mass = node.l_mass
                    node.l_mass = 0
            self.counter = 0

        return self

    def score_one(self, x):
        score = 0

        for tree in self.trees:
            for depth, node in enumerate(tree.path(x)):
                if node.r_mass < self.size_limit:
                    break
            score += node.r_mass * 2 ** depth

        return score / self._max_score
