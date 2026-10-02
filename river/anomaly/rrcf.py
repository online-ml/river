from __future__ import annotations

import collections
import random

from river import base
from river.tree.base import Branch, Leaf

__all__ = ["RobustRandomCutForest"]

_QUERY_INDEX = -1


class RobustRandomCutBranch(Branch):
    def __init__(self, left, right, feature, threshold, n_points, lower, upper):
        super().__init__(left, right)
        self.feature = feature
        self.threshold = threshold
        self.n_points = n_points
        self.lower = lower
        self.upper = upper
        self.parent: RobustRandomCutBranch | None = None

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    def replace_child(self, old_child, new_child):
        self.children = tuple(new_child if child is old_child else child for child in self.children)

    def next(self, x):
        if x[self.feature] <= self.threshold:
            return self.left
        return self.right

    def most_common_path(self):
        raise NotImplementedError

    @property
    def repr_split(self):
        return f"{self.feature} <= {self.threshold:.5f}"


class RobustRandomCutLeaf(Leaf):
    def __init__(self, index, parent, point, n_points=1):
        super().__init__(index=index, parent=parent, point=point, n_points=n_points)

    @property
    def lower(self):
        return self.point

    @property
    def upper(self):
        return self.point

    def __repr__(self):
        return f"Leaf({self.index})"


class RobustRandomCutTree:
    """A single robust random cut tree, grown one point at a time.

    The incremental `insert`/`forget`/`collusive_displacement` logic is adapted from the
    reference implementation of the algorithm, the `rrcf` package
    (https://github.com/kLabUM/rrcf, MIT license).

    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.root: RobustRandomCutBranch | RobustRandomCutLeaf | None = None
        self.leaves: dict[int, RobustRandomCutLeaf] = {}
        self.features: list[base.typing.FeatureName] | None = None

    def insert(self, point, index):
        if self.root is None:
            leaf = RobustRandomCutLeaf(index=index, parent=None, point=point)
            self.root = leaf
            self.features = list(point)
            self.leaves[index] = leaf
            return

        duplicate = self._find_duplicate(point)
        if duplicate is not None:
            self._update_counts_upwards(duplicate, step=1)
            self.leaves[index] = duplicate
            return

        node = self.root
        parent = None
        leaf = None
        new_branch = None
        for _ in range(len(self.leaves) + 1):
            feature, threshold = self._draw_cut(point, node)
            if threshold <= node.lower[feature]:
                leaf = RobustRandomCutLeaf(index=index, parent=None, point=point)
                new_branch = RobustRandomCutBranch(
                    left=leaf,
                    right=node,
                    feature=feature,
                    threshold=threshold,
                    n_points=leaf.n_points + node.n_points,
                    lower=None,
                    upper=None,
                )
                break
            elif threshold >= node.upper[feature]:
                leaf = RobustRandomCutLeaf(index=index, parent=None, point=point)
                new_branch = RobustRandomCutBranch(
                    left=node,
                    right=leaf,
                    feature=feature,
                    threshold=threshold,
                    n_points=leaf.n_points + node.n_points,
                    lower=None,
                    upper=None,
                )
                break
            else:
                parent = node
                node = node.next(point)

        if new_branch is None:
            raise RuntimeError("A cut separating the new point was not found.")

        node.parent = new_branch
        leaf.parent = new_branch
        new_branch.parent = parent
        if parent is not None:
            parent.replace_child(node, new_branch)
        else:
            self.root = new_branch
        self._update_counts_upwards(parent, step=1)
        self._tighten_bounds_upwards(new_branch)
        self.leaves[index] = leaf

    def forget(self, index):
        leaf = self.leaves.pop(index)

        if leaf.n_points > 1:
            self._update_counts_upwards(leaf, step=-1)
            return

        if leaf is self.root:
            self.root = None
            self.features = None
            return

        parent = leaf.parent
        sibling = parent.right if leaf is parent.left else parent.left

        if parent is self.root:
            sibling.parent = None
            self.root = sibling
            return

        grandparent = parent.parent
        sibling.parent = grandparent
        grandparent.replace_child(parent, sibling)
        self._update_counts_upwards(grandparent, step=-1)
        self._relax_bounds_upwards(grandparent, leaf.point)

    def collusive_displacement(self, index):
        leaf = self.leaves[index]
        if leaf is self.root:
            return 0.0
        node = leaf
        displacements = []
        while node.parent is not None:
            parent = node.parent
            sibling = parent.right if node is parent.left else parent.left
            displacements.append(sibling.n_points / node.n_points)
            node = parent
        return max(displacements)

    def _find_duplicate(self, point):
        node = self.root
        while isinstance(node, RobustRandomCutBranch):
            node = node.next(point)
        if node.point == point:
            return node
        return None

    def _update_counts_upwards(self, node, step):
        while node is not None:
            node.n_points += step
            node = node.parent

    def _draw_cut(self, point, node):
        cumulative_spans = []
        total_span = 0.0
        for feature in self.features:
            span = max(node.upper[feature], point[feature]) - min(
                node.lower[feature], point[feature]
            )
            total_span += span
            cumulative_spans.append(total_span)
        draw = self.rng.uniform(0, total_span)
        for feature, cumulative_span in zip(self.features, cumulative_spans):
            if cumulative_span >= draw:
                extended_lower = min(node.lower[feature], point[feature])
                return feature, extended_lower + cumulative_span - draw
        raise RuntimeError("No feature was selected for the cut.")

    def _merged_bounds(self, branch):
        lower = {}
        upper = {}
        for feature in self.features:
            lower[feature] = min(branch.left.lower[feature], branch.right.lower[feature])
            upper[feature] = max(branch.left.upper[feature], branch.right.upper[feature])
        return lower, upper

    def _tighten_bounds_upwards(self, branch):
        lower, upper = self._merged_bounds(branch)
        branch.lower = lower
        branch.upper = upper
        node = branch.parent
        while node is not None:
            changed = False
            for feature in self.features:
                if lower[feature] < node.lower[feature]:
                    node.lower[feature] = lower[feature]
                    changed = True
                if upper[feature] > node.upper[feature]:
                    node.upper[feature] = upper[feature]
                    changed = True
            if not changed:
                break
            node = node.parent

    def _relax_bounds_upwards(self, node, point):
        while node is not None:
            on_boundary = any(
                node.lower[feature] == point[feature] or node.upper[feature] == point[feature]
                for feature in self.features
            )
            if not on_boundary:
                break
            node.lower, node.upper = self._merged_bounds(node)
            node = node.parent


class RobustRandomCutForest(base.AnomalyDetector):
    """Robust Random Cut Forest (RRCF).

    An online anomaly detector built from an ensemble of robust random cut trees. Each tree keeps
    a bounded, sliding sample of the stream (a reservoir of the most recent `tree_size` points).
    A point's anomaly score is its *collusive displacement* (CoDisp): roughly, the expected change
    in model complexity caused by inserting the point, which is large for points that would move
    many others when added. The score returned by `score_one` is the collusive displacement
    averaged over the trees. Higher scores indicate more anomalous points.

    Cuts are drawn in proportion to the span of each feature within the bounding box of the sample,
    which makes the forest scale-aware and robust to irrelevant dimensions: a feature only receives
    cuts in proportion to how much it stretches the data.

    The implementation is dictionary based. The set of features is established from the first
    observation seen by `learn_one` (its keys, sorted). For subsequent points, any feature missing
    from an observation is treated as `0.0`, and any feature not seen in that first observation is
    ignored. RRCF therefore expects a consistent set of finite-valued features.

    Because cuts are span-proportional, RRCF is sensitive to the relative scale of the features: a
    feature with a much larger range will receive most of the cuts and dominate the score. Pairing
    the forest with `preprocessing.StandardScaler` is recommended so that all features contribute
    comparably. Avoid `preprocessing.MinMaxScaler` here: scoring a point before learning it yields
    an ill-defined transform on a cold start, and monotonic features (such as a timestamp) end up
    dominating the cuts.

    The anomaly score is unbounded and only meaningful relative to the forest, so evaluate it with a
    rank-based, scale-invariant metric such as `metrics.RollingROCAUC`. The threshold-based
    `metrics.ROCAUC` assumes scores in `[0, 1]` and will degenerate on these scores.

    Note that high scores indicate anomalies, whereas low scores indicate normal observations.

    Parameters
    ----------
    n_trees
        Number of trees in the forest.
    tree_size
        Maximum number of points held by each tree. Once a tree is full, the oldest point is
        forgotten before a new one is inserted, so each tree tracks a sliding window of the stream.
    seed
        Random number seed. A given `seed` and stream always produce the same scores.

    Attributes
    ----------
    trees
        The list of `RobustRandomCutTree` instances making up the forest.

    Examples
    --------

    >>> import random
    >>> from river import anomaly

    >>> rng = random.Random(42)
    >>> rrcf = anomaly.RobustRandomCutForest(n_trees=20, tree_size=64, seed=42)

    >>> for _ in range(200):
    ...     x = {"a": rng.gauss(0, 1), "b": rng.gauss(0, 1)}
    ...     rrcf.learn_one(x)

    A point drawn from the same distribution scores much lower than a distant outlier:

    >>> normal = rrcf.score_one({"a": 0.0, "b": 0.0})
    >>> anomalous = rrcf.score_one({"a": 8.0, "b": 8.0})
    >>> normal < anomalous
    True
    >>> round(anomalous, 4)
    51.025

    The example below combines RRCF with a `StandardScaler` in a pipeline and scores a real
    stream, using the rank-based `RollingROCAUC` so that the unbounded scores are handled correctly.

    >>> from river import compose
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import preprocessing

    >>> model = compose.Pipeline(
    ...     preprocessing.StandardScaler(),
    ...     anomaly.RobustRandomCutForest(n_trees=15, tree_size=100, seed=42)
    ... )

    >>> auc = metrics.RollingROCAUC()

    >>> for x, y in datasets.CreditCard().take(1000):
    ...     score = model.score_one(x)
    ...     model.learn_one(x)
    ...     auc.update(y, score)

    >>> auc
    RollingROCAUC: 95.39%

    References
    ----------
    [^1]: Guha, S., Mishra, N., Roy, G. and Schrijvers, O., 2016, June. Robust random cut forest based anomaly detection on streams. In International conference on machine learning (pp. 2712-2721). PMLR. http://proceedings.mlr.press/v48/guha16.pdf
    [^2]: Bartos, M., Mullapudi, A. and Troutman, S., 2019. rrcf: Implementation of the Robust Random Cut Forest algorithm for anomaly detection on streams. Journal of Open Source Software, 4(35), p.1336. https://github.com/kLabUM/rrcf

    """

    def __init__(
        self,
        n_trees: int = 40,
        tree_size: int = 256,
        seed: int | None = None,
    ):
        self.n_trees = n_trees
        self.tree_size = tree_size
        self.seed = seed
        self.rng = random.Random(seed)

        self.trees: list[RobustRandomCutTree] = [
            RobustRandomCutTree(rng=self.rng) for _ in range(n_trees)
        ]
        self._feature_names: list[base.typing.FeatureName] | None = None
        self._index = 0
        self._index_window: collections.deque = collections.deque()

    def _canonicalize(self, x: dict) -> dict:
        assert self._feature_names is not None
        return {feature: float(x.get(feature, 0.0)) for feature in self._feature_names}

    def learn_one(self, x: dict) -> None:
        if self._feature_names is None:
            self._feature_names = sorted(x.keys())

        point = self._canonicalize(x)
        index = self._index
        self._index += 1

        if len(self._index_window) >= self.tree_size:
            oldest = self._index_window.popleft()
            for member in self.trees:
                member.forget(oldest)

        for member in self.trees:
            member.insert(point, index)

        self._index_window.append(index)

    def score_one(self, x: dict) -> float:
        if self._feature_names is None or not self._index_window:
            return 0.0

        point = self._canonicalize(x)
        state = self.rng.getstate()
        total = 0.0
        for member in self.trees:
            member.insert(point, _QUERY_INDEX)
            total += member.collusive_displacement(_QUERY_INDEX)
            member.forget(_QUERY_INDEX)
        self.rng.setstate(state)

        return total / len(self.trees)

    @classmethod
    def _unit_test_params(cls):
        yield {"n_trees": 5, "tree_size": 32}
