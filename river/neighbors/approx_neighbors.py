from __future__ import annotations

import functools
import statistics
import typing

from river import base, utils
from river.neighbors.approx import SWINN


def distance_wrapper(item1, item2, dist):
    return dist(item1[0], item2[0])


class ANNClassifier(base.Classifier):
    """Approximate Nearest Neighbor Classifier.

    This implementation relies on SWINN to keep a sliding window of the most recent data.
    Search queries are approximate, considering the graph-based nearest neighbor search index.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last observed samples.
    warm_up
        Number of instances to buffer to store before switching to the approximate search.
        Before `warm_up` instances are observed, search will be exact and exhaustive.
    distance_func
    graph_k
        The number of neighbors used to build the graph-based search index.
    max_candidates
        The maximum number of candidates used in SWINN's neighborhood joins.
    delta
        Early stop parameter in SWINN's neighborhood refinement procedure.
    prune_prob
        The probability of pruning potentially redundant edges in SWINN's search graph.
    n_iters
        The maximum number of neighborhood join iterations performed by SWINN to refine the search graph.
    epsilon
        Distance bound used in SWINN to aid in avoiding local minima while traversing the search graph.
        The higher its value, the most accurate the search queries become, at the cost of increased running time.
    weighted
        Weight the contribution of each neighbor by it's inverse distance.
    seed
        Random seed for reproducibility.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.ANNClassifier()
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metrics.Accuracy())
    Accuracy: 84.55%


    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        warm_up: int = 500,
        distance_func: typing.Callable = None,
        graph_k: int = 20,
        max_candidates: int = 50,
        delta: float = 0.001,
        prune_prob: float = 0.0,
        n_iters: int = 10,
        epsilon: float = 0.1,
        weighted: bool = False,
        seed: int = None,
    ):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.warm_up = warm_up
        self.distance_func = (
            functools.partial(utils.math.minkowski_distance, p=2)
            if distance_func is None
            else distance_func
        )
        self.graph_k = graph_k if graph_k else 2 * self.n_neighbors
        self.max_candidates = max_candidates
        self.delta = delta
        self.prune_prob = prune_prob
        self.n_iters = n_iters
        self.epsilon = epsilon

        self.weighted = weighted
        self.seed = seed

        distance_func = functools.partial(distance_wrapper, dist=self.distance_func)
        self._buffer = SWINN(
            n_neighbors=self.graph_k,
            maxlen=self.window_size,
            warm_up=self.warm_up,
            dist_func=distance_func,
            max_candidates=self.max_candidates,
            delta=self.delta,
            prune_prob=self.prune_prob,
            n_iters=self.n_iters,
            seed=self.seed,
        )

        self.classes_: set[base.typing.ClfTarget] = set()

    def learn_one(self, x, y):
        self.classes_.add(y)
        self._buffer.add((x, y))

    def predict_proba_one(self, x):
        proba = {cid: 0.0 for cid in self.classes_}
        if len(self._buffer) == 0:
            return proba

        points, dists = self._buffer.search((x,), self.n_neighbors, epsilon=self.epsilon)

        # If the closest neighbor has a distance of 0, then return it's output
        if dists[0] == 0:
            proba[points[0]][1] = 1.0
            return proba

        if not self.weighted:  # Uniform weights
            for point in points:
                proba[point[1]] += 1.0
        else:  # Use the inverse of the distance to weight the votes
            for d, index in zip(dists, points):
                proba[point[1]] += 1.0 / d

        return utils.math.softmax(proba)


class ANNRegressor(base.Regressor):
    def __init__(
        self,
        k: int = 5,
        window_size: int = 1000,
        warm_up: int = 100,
        dist_func: typing.Callable = None,
        graph_k: int = None,
        max_candidates: int = None,
        delta: float = 0.001,
        prune_prob: float = 1.0,
        n_iters: int = 10,
        epsilon: float = 0.1,
        weighted: bool = False,
        seed: int = None,
    ):
        self.k = k
        self.window_size = window_size
        self.warm_up = warm_up
        self.dist_func = (
            functools.partial(utils.math.minkowski_distance, p=2)
            if dist_func is None
            else dist_func
        )
        self.graph_k = graph_k if graph_k else 2 * self.k
        self.max_candidates = max_candidates
        self.delta = delta
        self.prune_prob = prune_prob
        self.n_iters = n_iters
        self.epsilon = epsilon

        self.weighted = weighted
        self.seed = seed

        dist_func = functools.partial(distance_wrapper, dist=self.dist_func)
        self._buffer = SWINN(
            n_neighbors=self.graph_k,
            maxlen=self.window_size,
            warm_up=self.warm_up,
            dist_func=dist_func,
            max_candidates=self.max_candidates,
            delta=self.delta,
            prune_prob=self.prune_prob,
            n_iters=self.n_iters,
            seed=self.seed,
        )

    def learn_one(self, x, y):
        self._buffer.add((x, y))

    def predict_one(self, x):
        result = self._buffer.search((x,), k=self.k, epsilon=self.epsilon)

        if not result:
            return 0.0

        neighbors, dists = result

        if dists[0] == 0 and not neighbors[0][1] is not None:
            return neighbors[0][1]

        ys = [n[1] for n in neighbors if n[1] is not None]

        if len(ys) != len(dists):
            dists = [d for d, y in zip(dists, ys) if y is not None]

        sum_ = sum(1 / d for d in dists)

        if not self.weighted or sum_ == 0.0:
            return statistics.mean(ys)

        # weighted mean based on distance
        return sum(y / d for y, d in zip(ys, dists)) / sum_
