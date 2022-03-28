from typing import Callable

import numpy as np

from river import base

from .base_neighbors import NearestNeighbors


class KNNRegressor(NearestNeighbors, base.Regressor):
    """K-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last `window_size`
    training samples. Predictions are obtained by aggregating the values of the
    closest n_neighbors stored-samples with respect to a query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.

    window_size
        The maximum size of the window storing the last observed samples.

    aggregation_method
        The method to aggregate the target values of neighbors.
            | 'mean'
            | 'median'
            | 'weighted_mean'

    min_distance_keep
        The minimum distance (similarity) to consider adding a point to the window.
        E.g., a value of 0.0 will add even exact duplicates. Default is 0.05 to add
        similar but not exactly the same points.

    distance_func
        An optional distance function that should accept an a=, b=, and any
        custom set of kwargs (defined in distance_func_kwargs). If not defined,
        the default Minkowski distance is used.

    distance_func_kwargs
        A dictionary to pass as kwargs to your distance function, in addition
        to a= and b=. If distance_func is set to None, these are ignored,
        and are set to including the power parameter for the Minkowski metric.
        For this parameter, when `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.

    Notes
    -----
    See the NearestNeighbors documentation for details about the base model.

    Examples
    --------
    >>> from river import datasets, neighbors
    >>> from river import evaluate, metrics
    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     neighbors.KNNRegressor(window_size=50)
    ... )


    >>> for x, y in dataset.take(100):
    ...     model = model.learn_one(x, y)

    >>> for x, y in dataset.take(1):
    ...     model.predict_one(x)
    41.839342
    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _WEIGHTED_MEAN = "weighted_mean"

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        aggregation_method: str = "mean",
        min_distance_keep: float = 0.0,
        distance_func: Callable = None,
        distance_func_kwargs: dict = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            min_distance_keep=min_distance_keep,
            distance_func=distance_func,
            distance_func_kwargs=distance_func_kwargs,
        )
        if aggregation_method not in {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}:
            raise ValueError(
                "Invalid aggregation_method: {}.\n"
                "Valid options are: {}".format(
                    aggregation_method, {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}
                )
            )
        self.aggregation_method = aggregation_method

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}

    def predict_one(self, x):
        """Predict the target value of a set of features `x`.

        Search the window for the `n_neighbors` nearest neighbors. Return
        a default prediction if the size of the window is 0 (no neighbors yet)

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
            The prediction.
        """
        # Find the nearest neighbors!
        nearest = self.find_nearest(x=x, n_neighbors=self.n_neighbors)

        if not nearest:
            return 0.0

        # If the closest has a distance of 0 (it's the same) return it's output
        # BUT only if the output (y) is not None.
        if nearest[0][-1] == 0 and nearest[0][1] is not None:
            return nearest[0][1]

        # Only include neighbors in the sum that are non None
        neighbor_vals = [n[1] for n in nearest if n[1] is not None]

        if self.aggregation_method == self._MEAN:
            return np.mean(neighbor_vals)

        if self.aggregation_method == self._MEDIAN:
            return np.median(neighbor_vals)

        # weighted mean based on distance
        dists = [n[-1] for n in nearest if n[1] is not None]
        return sum(y / d for y, d in zip(neighbor_vals, dists)) / sum(
            1 / d for d in dists
        )
