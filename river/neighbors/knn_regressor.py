import functools
import statistics

from river import base, utils
from river.neighbors import NearestNeighbors
from river.neighbors.base import DistanceFunc, FunctionWrapper


class KNNRegressor(base.Regressor):
    """K-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last `window_size`
    training samples. Predictions are obtained by aggregating the values of the
    closest n_neighbors stored samples with respect to a query sample.

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
        E.g., a value of 0.0 will add even exact duplicates.
    distance_func
        An optional distance function that should accept an a=, b=, and any
        custom set of kwargs. If not defined, the Minkowski distance is used with
        p=2 (Euclidean distance). See the example section for more details.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = neighbors.KNNRegressor(window_size=50)
    >>> evaluate.progressive_val_score(dataset, model, metrics.RMSE())
    RMSE: 1.427746

    When defining a custom distance function you can rely on `functools.partial` to set default
    parameter values. For instance, let's use the Manhattan function instead of the default Euclidean distance:

    >>> import functools
    >>> from river.utils.math import minkowski_distance
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.KNNRegressor(
    ...         window_size=50,
    ...         distance_func=functools.partial(minkowski_distance, p=1)
    ...     )
    ... )
    >>> evaluate.progressive_val_score(dataset, model, metrics.RMSE())
    RMSE: 1.460385

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
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.min_distance_keep = min_distance_keep
        self.distance_func = (
            distance_func
            if distance_func is not None
            else functools.partial(utils.math.minkowski_distance, p=2)
        )

        self._nn = NearestNeighbors(
            window_size=self.window_size,
            min_distance_keep=min_distance_keep,
            distance_func=FunctionWrapper(self.distance_func),
        )

        self._check_aggregation_method(aggregation_method)
        self.aggregation_method = aggregation_method

    def _check_aggregation_method(self, method):
        """Ensure validation method is known to the model.

        Raises a ValueError if not.

        Parameters
        ----------
        method
            The suplied aggregration method.
        """
        if method not in {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}:
            raise ValueError(
                "Invalid aggregation_method: {}.\n"
                "Valid options are: {}".format(
                    method, {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}
                )
            )

    def learn_one(self, x, y):
        self._nn.update((x, y), n_neighbors=self.n_neighbors)
        return self

    def predict_one(self, x):
        # Find the nearest neighbors!
        nearest = self._nn.find_nearest((x, None), n_neighbors=self.n_neighbors)

        if not nearest:
            return 0.0

        # For each in nearest, call it 'item"
        # item[0] is the original item (x, y)
        # item[-1] is the distance

        # If the closest distance is 0 (it's the same) return it's output (y)
        # BUT only if the output (y) is not None.
        if nearest[0][-1] == 0 and nearest[0][0][1] is not None:
            return nearest[0][0][1]

        # Only include neighbors in the sum that are not None
        neighbor_vals = [n[0][1] for n in nearest if n[0][1] is not None]

        if self.aggregation_method == self._MEDIAN:
            return statistics.median(neighbor_vals)

        dists = [n[-1] for n in nearest if n[0][1] is not None]
        sum_ = sum(1 / d for d in dists)

        if self.aggregation_method == self._MEAN or sum_ == 0.0:
            return statistics.mean(neighbor_vals)

        # weighted mean based on distance
        return sum(y / d for y, d in zip(neighbor_vals, dists)) / sum_
