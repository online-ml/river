from __future__ import annotations

import collections
import random

import numpy as np

from river import anomaly

__all__ = ["RobustRandomCutForest"]

_QUERY_INDEX = -1


class _Branch:
    __slots__ = ["q", "p", "l", "r", "u", "n", "b"]

    def __init__(self, q, p, left=None, right=None, u=None, n=0, b=None):
        self.l = left
        self.r = right
        self.u = u
        self.q = q
        self.p = p
        self.n = n
        self.b = b

    def __repr__(self):
        return f"Branch(q={self.q}, p={self.p:.2f})"


class _Leaf:
    __slots__ = ["i", "d", "u", "x", "n", "b"]

    def __init__(self, i, d=None, u=None, x=None, n=1):
        self.u = u
        self.i = i
        self.d = d
        self.x = x
        self.n = n
        self.b = x.reshape(1, -1)

    def __repr__(self):
        return f"Leaf({self.i})"


class _RCTree:
    """A single robust random cut tree.

    This is a faithful port of the ``RCTree`` data structure from the ``rrcf`` package
    (https://github.com/kLabUM/rrcf, MIT license), restricted to the incremental streaming
    interface used by River: an empty tree grown one point at a time via ``insert_point`` and
    trimmed via ``forget_point``.
    """

    def __init__(self, random_state=None):
        if isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState()
        self.leaves: dict = {}
        self.root = None
        self.ndim = None

    def map_leaves(self, node, op=(lambda x: None), *args, **kwargs):
        if isinstance(node, _Branch):
            if node.l:
                self.map_leaves(node.l, op=op, *args, **kwargs)
            if node.r:
                self.map_leaves(node.r, op=op, *args, **kwargs)
        else:
            op(node, *args, **kwargs)

    def insert_point(self, point, index, tolerance=None):
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if self.root is None:
            leaf = _Leaf(x=point, i=index, d=0)
            self.root = leaf
            self.ndim = point.size
            self.leaves[index] = leaf
            return leaf
        if point.size != self.ndim:
            raise ValueError("Point must be same dimension as existing points in tree.")
        if index in self.leaves:
            raise KeyError("Index already exists in leaves dict.")
        duplicate = self.find_duplicate(point, tolerance=tolerance)
        if duplicate:
            self._update_leaf_count_upwards(duplicate, inc=1)
            self.leaves[index] = duplicate
            return duplicate
        node = self.root
        parent = node.u
        maxdepth = max([leaf.d for leaf in self.leaves.values()])
        depth = 0
        branch = None
        side = "l"
        for _ in range(maxdepth + 1):
            bbox = node.b
            cut_dimension, cut = self._insert_point_cut(point, bbox)
            if cut <= bbox[0, cut_dimension]:
                leaf = _Leaf(x=point, i=index, d=depth)
                branch = _Branch(q=cut_dimension, p=cut, left=leaf, right=node, n=(leaf.n + node.n))
                break
            elif cut >= bbox[-1, cut_dimension]:
                leaf = _Leaf(x=point, i=index, d=depth)
                branch = _Branch(q=cut_dimension, p=cut, left=node, right=leaf, n=(leaf.n + node.n))
                break
            else:
                depth += 1
                if point[node.q] <= node.p:
                    parent = node
                    node = node.l
                    side = "l"
                else:
                    parent = node
                    node = node.r
                    side = "r"
        if branch is None:
            raise AssertionError("Error with program logic: a cut was not found.")
        node.u = branch
        leaf.u = branch
        branch.u = parent
        if parent is not None:
            setattr(parent, side, branch)
        else:
            self.root = branch
        self.map_leaves(branch, op=self._increment_depth, inc=1)
        self._update_leaf_count_upwards(parent, inc=1)
        self._tighten_bbox_upwards(branch)
        self.leaves[index] = leaf
        return leaf

    def forget_point(self, index):
        leaf = self.leaves[index]
        if leaf.n > 1:
            self._update_leaf_count_upwards(leaf, inc=-1)
            return self.leaves.pop(index)
        if leaf is self.root:
            self.root = None
            self.ndim = None
            return self.leaves.pop(index)
        parent = leaf.u
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        if parent is self.root:
            del parent
            sibling.u = None
            self.root = sibling
            if isinstance(sibling, _Leaf):
                sibling.d = 0
            else:
                self.map_leaves(sibling, op=self._increment_depth, inc=-1)
            return self.leaves.pop(index)
        grandparent = parent.u
        sibling.u = grandparent
        if parent is grandparent.l:
            grandparent.l = sibling
        else:
            grandparent.r = sibling
        parent = grandparent
        self.map_leaves(sibling, op=self._increment_depth, inc=-1)
        self._update_leaf_count_upwards(parent, inc=-1)
        point = leaf.x
        self._relax_bbox_upwards(parent, point)
        return self.leaves.pop(index)

    def _update_leaf_count_upwards(self, node, inc=1):
        while node:
            node.n += inc
            node = node.u

    def query(self, point, node=None):
        if not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.ravel()
        if node is None:
            node = self.root
        return self._query(point, node)

    def _query(self, point, node):
        if isinstance(node, _Leaf):
            return node
        else:
            if point[node.q] <= node.p:
                return self._query(point, node.l)
            else:
                return self._query(point, node.r)

    def find_duplicate(self, point, tolerance=None):
        nearest = self.query(point)
        if tolerance is None:
            if (nearest.x == point).all():
                return nearest
        else:
            if np.isclose(nearest.x, point, rtol=tolerance).all():
                return nearest
        return None

    def disp(self, leaf):
        if not isinstance(leaf, _Leaf):
            leaf = self.leaves[leaf]
        if leaf is self.root:
            return 0
        parent = leaf.u
        if leaf is parent.l:
            sibling = parent.r
        else:
            sibling = parent.l
        return sibling.n

    def codisp(self, leaf):
        if not isinstance(leaf, _Leaf):
            leaf = self.leaves[leaf]
        if leaf is self.root:
            return 0
        node = leaf
        results = []
        for _ in range(node.d):
            parent = node.u
            if parent is None:
                break
            if node is parent.l:
                sibling = parent.r
            else:
                sibling = parent.l
            num_deleted = node.n
            displacement = sibling.n
            result = displacement / num_deleted
            results.append(result)
            node = parent
        co_displacement = max(results)
        return co_displacement

    def get_bbox(self, branch=None):
        if branch is None:
            branch = self.root
        mins = np.full(self.ndim, np.inf)
        maxes = np.full(self.ndim, -np.inf)
        self.map_leaves(branch, op=self._get_bbox, mins=mins, maxes=maxes)
        bbox = np.vstack([mins, maxes])
        return bbox

    def _lr_branch_bbox(self, node):
        bbox = np.vstack(
            [
                np.minimum(node.l.b[0, :], node.r.b[0, :]),
                np.maximum(node.l.b[-1, :], node.r.b[-1, :]),
            ]
        )
        return bbox

    def _get_bbox(self, x, mins, maxes):
        lt = x.x < mins
        gt = x.x > maxes
        mins[lt] = x.x[lt]
        maxes[gt] = x.x[gt]

    def _tighten_bbox_upwards(self, node):
        bbox = self._lr_branch_bbox(node)
        node.b = bbox
        node = node.u
        while node:
            lt = bbox[0, :] < node.b[0, :]
            gt = bbox[-1, :] > node.b[-1, :]
            lt_any = lt.any()
            gt_any = gt.any()
            if lt_any or gt_any:
                if lt_any:
                    node.b[0, :][lt] = bbox[0, :][lt]
                if gt_any:
                    node.b[-1, :][gt] = bbox[-1, :][gt]
            else:
                break
            node = node.u

    def _relax_bbox_upwards(self, node, point):
        while node:
            bbox = self._lr_branch_bbox(node)
            if not ((node.b[0, :] == point) | (node.b[-1, :] == point)).any():
                break
            node.b[0, :] = bbox[0, :]
            node.b[-1, :] = bbox[-1, :]
            node = node.u

    def _increment_depth(self, x, inc=1):
        x.d += inc

    def _insert_point_cut(self, point, bbox):
        bbox_hat = np.empty(bbox.shape)
        bbox_hat[0, :] = np.minimum(bbox[0, :], point)
        bbox_hat[-1, :] = np.maximum(bbox[-1, :], point)
        b_span = bbox_hat[-1, :] - bbox_hat[0, :]
        b_range = b_span.sum()
        r = self.rng.uniform(0, b_range)
        span_sum = np.cumsum(b_span)
        cut_dimension = np.inf
        for j in range(len(span_sum)):
            if span_sum[j] >= r:
                cut_dimension = j
                break
        if not np.isfinite(cut_dimension):
            raise ValueError("Cut dimension is not finite.")
        cut = bbox_hat[0, cut_dimension] + span_sum[cut_dimension] - r
        return cut_dimension, cut


class RobustRandomCutForest(anomaly.base.AnomalyDetector):
    """Robust Random Cut Forest (RRCF).

    An online anomaly detector built from an ensemble of robust random cut trees. Each tree keeps
    a bounded, sliding sample of the stream (a reservoir of the most recent `tree_size` points).
    A point's anomaly score is its *collusive displacement* (CoDisp): roughly, the expected change
    in model complexity caused by inserting the point, which is large for points that would move
    many others when added. The score returned by `score_one` is the CoDisp averaged over the
    trees. Higher scores indicate more anomalous points.

    Cuts are drawn in proportion to the span of each feature within the bounding box of the sample,
    which makes the forest scale-aware and robust to irrelevant dimensions: a feature only receives
    cuts in proportion to how much it stretches the data.

    The public API is dictionary based. A fixed feature ordering is established from the first
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
        Random number seed. Each tree is given its own seed derived from this value, so that a
        given `seed` and stream always produce the same scores.

    Attributes
    ----------
    trees
        The list of `_RCTree` instances making up the forest.

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
    53.5494

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
    RollingROCAUC: 95.64%

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

        seeder = random.Random(seed)
        self.trees: list[_RCTree] = [
            _RCTree(random_state=seeder.randint(0, 2**32 - 1)) for _ in range(n_trees)
        ]
        self._feature_names: list | None = None
        self._index = 0
        self._index_window: collections.deque = collections.deque()

    def _to_vector(self, x: dict) -> np.ndarray:
        assert self._feature_names is not None
        return np.array(
            [float(x.get(feature, 0.0)) for feature in self._feature_names],
            dtype=np.float64,
        )

    def learn_one(self, x: dict) -> None:
        if self._feature_names is None:
            self._feature_names = sorted(x.keys())

        point = self._to_vector(x)
        index = self._index
        self._index += 1

        if len(self._index_window) >= self.tree_size:
            oldest = self._index_window.popleft()
            for tree in self.trees:
                tree.forget_point(oldest)

        for tree in self.trees:
            tree.insert_point(point, index)

        self._index_window.append(index)

    def score_one(self, x: dict) -> float:
        if self._feature_names is None or not self._index_window:
            return 0.0

        point = self._to_vector(x)
        total = 0.0
        for tree in self.trees:
            state = tree.rng.get_state()
            tree.insert_point(point, _QUERY_INDEX)
            total += float(tree.codisp(_QUERY_INDEX))
            tree.forget_point(_QUERY_INDEX)
            tree.rng.set_state(state)

        return total / len(self.trees)

    @classmethod
    def _unit_test_params(cls):
        yield {"n_trees": 5, "tree_size": 32}
