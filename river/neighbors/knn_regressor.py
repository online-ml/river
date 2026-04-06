from __future__ import annotations

import statistics

from river import base, utils
from river.neighbors import SWINN

from .base import (
    BaseNN,
    FunctionWrapper,
    _euclidean_tuple_distance,  # type: ignore[attr-defined]
)


class KNNRegressor(base.Regressor):
    """K-Nearest Neighbors regressor.

    Samples are stored using a first-in, first-out strategy. The strategy to perform search
    queries in the data buffer is defined by the `engine` parameter. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored samples with respect to a query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    engine
        The search engine used to store the instances and perform search queries. Depending
        on the choose engine, search will be exact or approximate. Please, consult the
        documentation of each available search engine for more details on its usage.
        By default, use the `SWINN` search engine for approximate search queries.
    aggregation_method
        The method to aggregate the target values of neighbors.
            | 'mean'
            | 'median'
            | 'weighted_mean'

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = neighbors.KNNRegressor()
    >>> evaluate.progressive_val_score(dataset, model, metrics.RMSE())
    RMSE: 1.427743

    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _WEIGHTED_MEAN = "weighted_mean"

    def __init__(
        self,
        n_neighbors: int = 5,
        engine: BaseNN | None = None,
        aggregation_method: str = "mean",
    ):
        self.n_neighbors = n_neighbors

        _default_dist = utils.math._euclidean_distance  # type: ignore[attr-defined]
        if engine is None:
            engine = SWINN(dist_func=_default_dist)  # type: ignore[arg-type]

        if not isinstance(engine.dist_func, FunctionWrapper):
            if engine.dist_func is _default_dist:
                engine.dist_func = _euclidean_tuple_distance  # type: ignore[assignment]
            elif engine.dist_func is not _euclidean_tuple_distance:
                engine.dist_func = FunctionWrapper(engine.dist_func)

        self.engine = engine

        # Create a fresh copy of the supplied search engine
        self._nn: BaseNN = self.engine.clone(include_attributes=True)

        self._check_aggregation_method(aggregation_method)
        self.aggregation_method = aggregation_method

    @classmethod
    def _unit_test_params(cls):
        from river.neighbors import LazySearch

        yield {
            "n_neighbors": 3,
            "engine": LazySearch(
                window_size=50,
                dist_func=utils.math._euclidean_distance,  # type: ignore[attr-defined]
            ),
        }

    def _check_aggregation_method(self, method):
        """Ensure validation method is known to the model.

        Raises a ValueError if not.

        Parameters
        ----------
        method
            The supplied aggregation method.
        """
        if method not in {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}:
            raise ValueError(
                f"Invalid aggregation_method: {method}.\n"
                f"Valid options are: {(self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN)}"
            )

    def learn_one(self, x, y):
        self._nn.append((x, y))

    def predict_one(self, x, **kwargs):
        neighbors, distances = self._nn.search((x, None), n_neighbors=self.n_neighbors, **kwargs)

        if not neighbors:
            return 0.0
        # If the closest distance is 0 (it's the same) return it's output (y)
        if distances[0] == 0:
            return neighbors[0][1]

        neighbor_vals = [n[1] for n in neighbors]

        if self.aggregation_method == self._MEDIAN:
            return statistics.median(neighbor_vals)

        sum_ = sum(1 / d for d in distances)
        if self.aggregation_method == self._MEAN or sum_ == 0.0:
            return statistics.mean(neighbor_vals)

        # weighted mean based on distance
        return sum(y / d for y, d in zip(neighbor_vals, distances)) / sum_
