from __future__ import annotations

import collections
import functools
import math
import random
import typing

from river import base
from river.anomaly.hst import HSTBranch, HSTLeaf, make_padded_tree

__all__ = ["RBFSampler", "RandomTreesEmbedding"]


class RBFSampler(base.Transformer):
    """Extracts random features which approximate an RBF kernel.

    This is a powerful way to give non-linear capacity to linear classifiers. This method is also
    called "random Fourier features" in the literature.

    Parameters
    ----------
    gamma
        RBF kernel parameter in `(-gamma * x^2)`.
    n_components
        Number of samples per original feature. Equals the dimensionality of the computed feature
        space.
    seed
        Random number seed.

    Examples
    --------

    >>> from river import feature_extraction as fx
    >>> from river import linear_model as lm
    >>> from river import optim
    >>> from river import stream

    >>> # XOR function
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> Y = [0, 0, 1, 1]

    >>> model = lm.LogisticRegression(optimizer=optim.SGD(.1))

    >>> for x, y in stream.iter_array(X, Y):
    ...     model.learn_one(x, y)
    ...     y_pred = model.predict_one(x)
    ...     print(y, int(y_pred))
    0 0
    0 0
    1 0
    1 1

    >>> model = (
    ...     fx.RBFSampler(seed=3) |
    ...     lm.LogisticRegression(optimizer=optim.SGD(.1))
    ... )

    >>> for x, y in stream.iter_array(X, Y):
    ...     model.learn_one(x, y)
    ...     y_pred = model.predict_one(x)
    ...     print(y, int(y_pred))
    0 0
    0 0
    1 1
    1 1

    References
    ----------
    [^1]: [Rahimi, A. and Recht, B., 2008. Random features for large-scale kernel machines. In Advances in neural information processing systems (pp. 1177-1184](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

    """

    def __init__(self, gamma=1.0, n_components=100, seed: int | None = None):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights: collections.defaultdict[typing.Hashable, typing.Callable] = (
            collections.defaultdict(self._random_weights)
        )
        self.offsets = [self.rng.uniform(0, 2 * math.pi) for _ in range(n_components)]

    def _random_weights(self):
        return [
            math.sqrt(2 * self.gamma) * self.rng.gauss(mu=0, sigma=1)
            for _ in range(self.n_components)
        ]

    def transform_one(self, x, y=None):
        return {
            (i, j): math.cos(xi * wj + self.offsets[j])
            for i, xi in x.items()
            for j, wj in enumerate(self.weights[i])
        }


class RandomTreesEmbedding(base.Transformer):
    """Embed samples according to the leaves of random trees.

    This transformer builds an ensemble of totally random trees and encodes each sample with the
    leaf reached in every tree. The output is sparse: exactly one binary feature is active per tree.

    This is the online counterpart of feeding a linear model with random-tree leaf indicators.
    The trees are built lazily from the first sample that is observed, either via `transform_one`
    or `learn_one`. If new features appear later in the stream, then the forest is rebuilt so that
    future splits may use them.

    Parameters
    ----------
    n_trees
        Number of trees in the ensemble.
    height
        Height of each tree. A tree of height `h` contains `2 ** h` leaves.
    limits
        Specifies the range of each feature. By default each feature is assumed to be in
        range `[0, 1]`.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> from river import feature_extraction as fx
    >>> from river import linear_model as lm
    >>> from river import optim

    >>> embedding = fx.RandomTreesEmbedding(n_trees=3, height=2, seed=42)
    >>> len(embedding.transform_one({'x': 0.3, 'y': 0.7}))
    3

    >>> model = (
    ...     fx.RandomTreesEmbedding(n_trees=5, height=3, seed=42) |
    ...     lm.LogisticRegression(optimizer=optim.SGD(0.1))
    ... )

    References
    ----------
    [^1]: [Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63(1), 3-42.](https://link.springer.com/article/10.1007/s10994-006-6226-1)
    [^2]: [scikit-learn random trees embedding](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html)

    """

    def __init__(
        self,
        n_trees=10,
        height=8,
        limits: dict[base.typing.FeatureName, tuple[float, float]] | None = None,
        seed: int | None = None,
    ):
        self.n_trees = n_trees
        self.height = height
        self.limits: collections.defaultdict = collections.defaultdict(
            functools.partial(tuple, (0.0, 1.0))
        )
        if limits is not None:
            self.limits.update(limits)
        self.seed = seed
        self.rng = random.Random(seed)

        self.trees: list[HSTBranch | HSTLeaf] = []
        self._leaf_indices: list[dict[int, int]] = []
        self._features: set[base.typing.FeatureName] = set()

    def _make_trees(self):
        self.trees = [
            make_padded_tree(
                limits={i: self.limits[i] for i in sorted(self._features)},
                height=self.height,
                padding=0.15,
                rng=self.rng,
                # kwargs
                r_mass=0,
                l_mass=0,
            )
            for _ in range(self.n_trees)
        ]

        self._leaf_indices = []
        for tree in self.trees:
            self._leaf_indices.append({id(leaf): i for i, leaf in enumerate(tree.iter_leaves())})

    def _ensure_initialized(self, x):
        new_features = set(x) - self._features
        if self.trees and not new_features:
            return

        self._features.update(x)
        if not self._features:
            return

        self._make_trees()

    def learn_one(self, x):
        self._ensure_initialized(x)

        for tree in self.trees:
            for node in tree.walk(x):
                node.l_mass += 1

    def transform_one(self, x):
        self._ensure_initialized(x)

        features = {}
        for tree_id, tree in enumerate(self.trees):
            leaf = tree if isinstance(tree, HSTLeaf) else tree.traverse(x)
            features[(tree_id, self._leaf_indices[tree_id][id(leaf)])] = 1.0

        return features
