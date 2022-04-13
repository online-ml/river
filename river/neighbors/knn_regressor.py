import statistics
from typing import Tuple

from river import base

from .base_neighbors import BaseKNN
from .neighbors import DistanceFunc


class KNNRegressor(BaseKNN, base.Regressor):
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

    Notes
    -----
    See the NearestNeighbors documentation for details about the base model,
    along with KNNBase for an example of providing your own distance function.

    Examples
    --------
    >>> from river import datasets, neighbors
    >>> from river import evaluate, metrics
    >>> dataset = datasets.TrumpApproval()

    >>> model = neighbors.KNNRegressor(window_size=50)
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
        distance_func: DistanceFunc = None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            min_distance_keep=min_distance_keep,
            distance_func=distance_func,
        )
        self._check_aggregation_method(aggregation_method)
        self.aggregation_method = aggregation_method

    def _check_aggregation_method(self, method):
        """Ensure validation method is known to the model.

        Raises a ValueError if not.

        Parameters
        ----------

        method
            The aggregration method as a string
        """
        if method not in {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}:
            raise ValueError(
                "Invalid aggregation_method: {}.\n"
                "Valid options are: {}".format(
                    method, {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}
                )
            )

    def learn_one(self, x, y=None, extra: [Tuple, list] = None):
        """Learn a set of features `x` and optional class `y`.
        Parameters:
            x: A dictionary of features.
            y: A class (optional if known).
            extra: an optional list or tuple of features to store
        Returns:
            self

        """
        self.nn.update((x, y), n_neighbors=self.n_neighbors, extra=extra)
        return self

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
        nearest = self.nn.find_nearest((x, None), n_neighbors=self.n_neighbors)

        if not nearest:
            return 0.0

        # For each in nearest, call it 'item"
        # item[0] is the original item (x, y)
        # item[-1] is the distance
        # item[1:n-1] are extra we don't use here

        # If the closest distance is 0 (it's the same) return it's output (y)
        # BUT only if the output (y) is not None.
        if nearest[0][-1] == 0 and nearest[0][0][1] is not None:
            return nearest[0][0][1]

        # Only include neighbors in the sum that are non None
        neighbor_vals = [n[0][1] for n in nearest if n[0][1] is not None]

        if self.aggregation_method == self._MEAN:
            return statistics.mean(neighbor_vals)

        if self.aggregation_method == self._MEDIAN:
            return statistics.median(neighbor_vals)

        # weighted mean based on distance
        dists = [n[-1] for n in nearest if n[0][1] is not None]
        return sum(y / d for y, d in zip(neighbor_vals, dists)) / sum(
            1 / d for d in dists
        )
