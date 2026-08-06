from __future__ import annotations

import math
import typing

from river import anomaly, utils
from river.neighbors import LazySearch
from river.neighbors.base import BaseNN, FunctionWrapper
from river.utils.vectordict import euclidean_distance_tuple as _euclidean_tuple_distance

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame

__all__ = ["LocalOutlierFactor"]


class LocalOutlierFactor(anomaly.base.AnomalyDetector):
    """Local Outlier Factor (LOF).

    The LOF of a sample measures how isolated it is relative to the density of its neighbors:
    a value around 1 means the sample lies in a region as dense as its neighborhood, while a
    value substantially above 1 flags an outlier sitting in a comparatively sparse region
    (Breunig et al., 2000).

    Samples are stored in a fixed-size sliding window managed by a nearest-neighbor `engine`
    (see `river.neighbors`). `learn_one` simply adds a sample to the window, so it runs in
    constant time and the memory footprint is bounded by the window size. `score_one` computes
    the LOF of a sample against the current window on demand, without modifying it. The natural
    streaming pattern is therefore to call `score_one` and then `learn_one` on each sample.

    Because scoring is done against the window only, the result matches the static LOF computed
    on those samples (e.g. scikit-learn's `LocalOutlierFactor(novelty=True)` fitted on the same
    window).

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors used to define the local neighborhood.
    engine
        The nearest-neighbor search engine, which stores the samples in a sliding window and
        answers neighbor queries. Defaults to `neighbors.LazySearch`, an exact brute-force
        search over the window. Pass `neighbors.SWINN` for approximate search, or a configured
        engine to control the `window_size` and the distance function.

    Examples
    --------

    >>> from river import anomaly

    >>> X = [0.5, 0.45, 0.43, 0.44, 0.445, 0.45, 0.0]
    >>> lof = anomaly.LocalOutlierFactor(n_neighbors=3)

    >>> for x in X[:3]:
    ...     lof.learn_one({"x": x})  # warming up

    >>> for x in X:
    ...     features = {"x": x}
    ...     print(f"Anomaly score for x={x:.3f}: {lof.score_one(features):.3f}")
    ...     lof.learn_one(features)
    Anomaly score for x=0.500: 0.929
    Anomaly score for x=0.450: 1.105
    Anomaly score for x=0.430: 1.044
    Anomaly score for x=0.440: 1.000
    Anomaly score for x=0.445: 0.944
    Anomaly score for x=0.450: 0.889
    Anomaly score for x=0.000: 36.111

    A mini-batch of samples can be learned at once from any
    [narwhals](https://github.com/narwhals-dev/narwhals)-compatible eager dataframe (pandas,
    polars, pyarrow, ...) with `learn_many`:

    >>> import pandas as pd
    >>> from river import datasets

    >>> rows = [x for x, _ in datasets.CreditCard().take(500)]
    >>> lof = anomaly.LocalOutlierFactor(n_neighbors=20)
    >>> lof.learn_many(pd.DataFrame(rows))
    >>> [round(lof.score_one(x), 3) for x in rows[:5]]
    [1.51, 1.355, 1.987, 1.566, 1.837]

    References
    ----------
    [^1]: Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander (2000).
          LOF: Identifying Density-Based Local Outliers. In: Proceedings of the 2000 ACM SIGMOD
          International Conference on Management of Data. 93-104. DOI: 10.1145/342009.335388.
    [^2]: David Pokrajac, Aleksandar Lazarevic, and Longin Jan Latecki (2007). Incremental Local
          Outlier Detection for Data Streams. In: Proceedings of the 2007 IEEE Symposium on
          Computational Intelligence and Data Mining (CIDM 2007). 504-515.
          DOI: 10.1109/CIDM.2007.368917.

    """

    def __init__(self, n_neighbors: int = 10, engine: BaseNN | None = None):
        self.n_neighbors = n_neighbors

        _default_dist = utils.math._euclidean_distance  # type: ignore[attr-defined]
        if engine is None:
            engine = LazySearch(window_size=1000, dist_func=_default_dist)  # type: ignore[arg-type]

        # Engage the Cython tuple fast-path when the default Euclidean distance is used.
        # Otherwise, wrap the user's distance so it reads the feature dict out of the stored tuple.
        if not isinstance(engine.dist_func, FunctionWrapper):
            if engine.dist_func is _default_dist:
                engine.dist_func = _euclidean_tuple_distance  # type: ignore[assignment]
            elif engine.dist_func is not _euclidean_tuple_distance:
                engine.dist_func = FunctionWrapper(engine.dist_func)

        self.engine = engine
        # Work on a fresh copy so the engine passed by the user is left untouched.
        self._nn: BaseNN = engine.clone(include_attributes=True)

    @classmethod
    def _unit_test_params(cls):
        # k=10 is a weak setting on the small, duplicate-heavy CreditCard check sample (so is
        # scikit-learn there); k=20 — scikit-learn's own default — is a competent, robust choice.
        yield {"n_neighbors": 20}

    def _unit_test_skips(self):
        # Scores depend on the float summation order of the features, so reordering them can flip
        # near-tied neighbors (as for KNNRegressor and the forest models).
        return {"check_shuffle_features_no_impact"}

    def learn_one(self, x: dict):
        # Copy x so the caller can safely mutate the input dict afterwards.
        self._nn.append((dict(x),))

    def learn_many(self, X: IntoDataFrame):
        """Update with a mini-batch of samples held in a dataframe.

        Any [narwhals](https://github.com/narwhals-dev/narwhals)-compatible eager dataframe
        (pandas, polars, pyarrow, ...) is accepted. Each row is added to the window in turn.

        Parameters
        ----------
        X
            A dataframe of samples.

        """
        for row in utils.dataframe.into_frame(X).iter_rows(named=True):
            self.learn_one(row)

    def score_one(self, x: dict) -> float:
        x = dict(x)
        neighbors, distances = self._query_neighborhood(x)
        if not neighbors:
            return 0.0

        # Every window point's neighborhood is constant while scoring a single sample, and the
        # same points recur as neighbors-of-neighbors, so memoize them for the duration.
        neighborhoods: dict[int, tuple[list, list]] = {}
        lrd_x = self._local_reachability_density(neighbors, distances, neighborhoods)
        lrd_neighbors = [
            self._local_reachability_density(
                *self._neighborhood(o[0], neighborhoods), neighborhoods
            )
            for o in neighbors
        ]
        score = sum(lrd_neighbors) / (len(lrd_neighbors) * lrd_x)
        # The window can be too small to assess a sample (e.g. a neighbor with no neighbors of
        # its own during warm-up), which leaves the score undefined. Treat that as "not anomalous".
        return score if math.isfinite(score) else 0.0

    def _query_neighborhood(self, x: dict) -> tuple[list, list]:
        """Return the neighbors of the scored sample `x` and their distances.

        A point is never its own neighbor: if `x` was already learned (or coincides exactly with
        a stored sample), the matching distance-0 entry is dropped so the score does not count
        `x` against itself. Otherwise all `n_neighbors` nearest window points are kept.
        """
        neighbors, distances = self._nn.search((x,), n_neighbors=self.n_neighbors + 1)
        if distances and distances[0] == 0.0:
            return neighbors[1:], distances[1:]
        return neighbors[: self.n_neighbors], distances[: self.n_neighbors]

    def _neighborhood(self, x: dict, cache: dict[int, tuple[list, list]]) -> tuple[list, list]:
        """Return a window point's neighbors and their distances, excluding the point itself.

        One extra neighbor is requested and the closest match — `x` paired with itself at
        distance 0 — is dropped. Results are memoized per scored sample by object identity.
        """
        key = id(x)
        if key not in cache:
            neighbors, distances = self._nn.search((x,), n_neighbors=self.n_neighbors + 1)
            cache[key] = (neighbors[1:], distances[1:])
        return cache[key]

    def _local_reachability_density(
        self, neighbors: list, distances: list, cache: dict[int, tuple[list, list]]
    ) -> float:
        """Inverse of the average reachability distance to a point's neighbors.

        The small additive constant mirrors scikit-learn: it keeps the density finite when a
        point coincides with its neighbors (all reachability distances zero).
        """
        if not neighbors:
            return float("inf")
        total = sum(max(d, self._k_distance(o[0], cache)) for o, d in zip(neighbors, distances))
        return 1.0 / (total / len(neighbors) + 1e-10)

    def _k_distance(self, x: dict, cache: dict[int, tuple[list, list]]) -> float:
        """Distance from `x` to its `n_neighbors`-th nearest neighbor in the window."""
        _, distances = self._neighborhood(x, cache)
        return distances[-1] if distances else 0.0
