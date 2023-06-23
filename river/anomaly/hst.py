from __future__ import annotations

import collections
import functools
import random

from river import anomaly, base, tree

__all__ = ["HalfSpaceTrees"]


class HSTBranch(tree.base.Branch):
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


class HSTLeaf(tree.base.Leaf):
    def __repr__(self):
        return str(self.r_mass)


def make_padded_tree(limits, height, padding, rng=random, **node_params):
    if height == 0:
        return HSTLeaf(**node_params)

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

    return HSTBranch(left=left, right=right, feature=on, threshold=at, **node_params)


class HalfSpaceTrees(anomaly.base.AnomalyDetector):
    """Half-Space Trees (HST).

    Half-space trees are an online variant of isolation forests. They work well when anomalies are
    spread out. However, they do not work well if anomalies are packed together in windows.

    By default, this implementation assumes that each feature has values that are comprised
    between 0 and 1. If this isn't the case, then you can manually specify the limits via the
    `limits` argument. If you do not know the limits in advance, then you can use a
    `preprocessing.MinMaxScaler` as an initial preprocessing step.

    The current implementation builds the trees the first time the `learn_one` method is called.
    Therefore, the first `learn_one` call might be slow, whereas subsequent calls will be very fast
    in comparison. In general, the computation time of both `learn_one` and `score_one` scales
    linearly with the number of trees, and exponentially with the height of each tree.

    Note that high scores indicate anomalies, whereas low scores indicate normal observations.

    Parameters
    ----------
    n_trees
        Number of trees to use.
    height
        Height of each tree. Note that a tree of height `h` is made up of `h + 1` levels and
        therefore contains `2 ** (h + 1) - 1` nodes.
    window_size
        Number of observations to use for calculating the mass at each node in each tree.
    limits
        Specifies the range of each feature. By default each feature is assumed to be in
        range `[0, 1]`.
    seed
        Random number seed.

    Examples
    --------

    >>> from river import anomaly

    >>> X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
    >>> hst = anomaly.HalfSpaceTrees(
    ...     n_trees=5,
    ...     height=3,
    ...     window_size=3,
    ...     seed=42
    ... )

    >>> for x in X[:3]:
    ...     hst = hst.learn_one({'x': x})  # Warming up

    >>> for x in X:
    ...     features = {'x': x}
    ...     hst = hst.learn_one(features)
    ...     print(f'Anomaly score for x={x:.3f}: {hst.score_one(features):.3f}')
    Anomaly score for x=0.500: 0.107
    Anomaly score for x=0.450: 0.071
    Anomaly score for x=0.430: 0.107
    Anomaly score for x=0.440: 0.107
    Anomaly score for x=0.445: 0.107
    Anomaly score for x=0.450: 0.071
    Anomaly score for x=0.000: 0.853

    The feature values are all comprised between 0 and 1. This is what is assumed by the model
    by default. In the following example, we construct a pipeline that scales the data online
    and ensures that the values of each feature are comprised between 0 and 1.

    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = compose.Pipeline(
    ...     preprocessing.MinMaxScaler(),
    ...     anomaly.HalfSpaceTrees(seed=42)
    ... )

    >>> auc = metrics.ROCAUC()

    >>> for x, y in datasets.CreditCard().take(2500):
    ...     score = model.score_one(x)
    ...     model = model.learn_one(x)
    ...     auc = auc.update(y, score)

    >>> auc
    ROCAUC: 91.15%

    You can also use the `evaluate.progressive_val_score` function to evaluate the model on a
    data stream.

    >>> from river import evaluate

    >>> model = model.clone()

    >>> evaluate.progressive_val_score(
    ...     dataset=datasets.CreditCard().take(2500),
    ...     model=model,
    ...     metric=metrics.ROCAUC(),
    ...     print_every=1000
    ... )
    [1,000] ROCAUC: 88.43%
    [2,000] ROCAUC: 89.28%
    [2,500] ROCAUC: 91.15%
    ROCAUC: 91.15%

    References
    ----------
    [^1]: [Tan, S.C., Ting, K.M. and Liu, T.F., 2011, June. Fast anomaly detection for streaming data. In Twenty-Second International Joint Conference on Artificial Intelligence.](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)

    """

    def __init__(
        self,
        n_trees=10,
        height=8,
        window_size=250,
        limits: dict[base.typing.FeatureName, tuple[float, float]] | None = None,
        seed: int | None = None,
    ):
        self.n_trees = n_trees
        self.window_size = window_size
        self.height = height
        self.limits: collections.defaultdict = collections.defaultdict(
            functools.partial(tuple, (0.0, 1.0))
        )
        if limits is not None:
            self.limits.update(limits)
        self.seed = seed
        self.rng = random.Random(seed)

        self.trees: list[HSTBranch] = []
        self.counter = 0
        self._first_window = True

    @property
    def size_limit(self):
        """This is the threshold under which the node search stops during the scoring phase.

        The value .1 is a magic constant indicated in the original paper.

        """
        return 0.1 * self.window_size

    @property
    def _max_score(self):
        """The largest potential anomaly score."""
        return self.n_trees * self.window_size * (2 ** (self.height + 1) - 1)

    def learn_one(self, x):
        # The trees are built when the first observation comes in
        if not self.trees:
            self.trees = [
                make_padded_tree(
                    limits={i: self.limits[i] for i in sorted(x)},
                    height=self.height,
                    padding=0.15,
                    rng=self.rng,
                    # kwargs
                    r_mass=0,
                    l_mass=0,
                )
                for _ in range(self.n_trees)
            ]

        # Update each tree
        for t in self.trees:
            for node in t.walk(x):
                node.l_mass += 1

        # Pivot the masses if necessary
        self.counter += 1
        if self.counter == self.window_size:
            for t in self.trees:
                for node in t.iter_dfs():
                    node.r_mass = node.l_mass
                    node.l_mass = 0
            self._first_window = False
            self.counter = 0

        return self

    def score_one(self, x):
        if self._first_window:
            return 0

        score = 0.0
        for t in self.trees:
            for depth, node in enumerate(t.walk(x)):
                score += node.r_mass * 2**depth
                if node.r_mass < self.size_limit:
                    break

        # Normalize the score between 0 and 1
        score /= self._max_score

        # We want high score -> anomaly, but we have high score -> normal
        return 1 - score
