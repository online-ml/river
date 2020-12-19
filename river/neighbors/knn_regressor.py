import numpy as np

from river import base
from river.utils import dict2numpy

from .base_neighbors import BaseNeighbors


class KNNRegressor(BaseNeighbors, base.Regressor):
    """k-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last
    `window_size` training samples. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored-samples with
    respect to a query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last observed samples.
    leaf_size
        scipy.spatial.cKDTree parameter. The maximum number of samples that can
        be stored in one leaf node, which determines from which point the algorithm
        will switch for a brute-force approach. The bigger this number the faster
        the tree construction time, but the slower the query time will be.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        Valid values are in the interval $[1, +\\infty)$
    aggregation_method
        The method to aggregate the target values of neighbors.
            | 'mean'
            | 'median'
            | 'weighted_mean'
    kwargs
        Other parameters passed to scipy.spatial.cKDTree.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.KNNRegressor(window_size=50)
    ... )

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 0.441308

    """

    _MEAN = "mean"
    _MEDIAN = "median"
    _WEIGHTED_MEAN = "weighted_mean"

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: float = 2,
        aggregation_method: str = "mean",
        **kwargs
    ):

        super().__init__(
            n_neighbors=n_neighbors, window_size=window_size, leaf_size=leaf_size, p=p, **kwargs
        )
        if aggregation_method not in {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}:
            raise ValueError(
                "Invalid aggregation_method: {}.\n"
                "Valid options are: {}".format(
                    aggregation_method, {self._MEAN, self._MEDIAN, self._WEIGHTED_MEAN}
                )
            )
        self.aggregation_method = aggregation_method
        self.kwargs = kwargs

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}

    def learn_one(self, x, y):
        """Update the model with a set of features `x` and a real target value `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A numeric target.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the `window_size` is reached, removing older results.

        """

        x_arr = dict2numpy(x)
        self.data_window.append(x_arr, y)

        return self

    def predict_one(self, x):
        """Predict the target value of a set of features `x`.

        Search the KDTree for the `n_neighbors` nearest neighbors.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
            The prediction.

        """

        if self.data_window.size == 0:
            # Not enough information available, return default prediction
            return 0.0

        x_arr = dict2numpy(x)

        dists, neighbor_idx = self._get_neighbors(x_arr)
        target_buffer = self.data_window.targets_buffer

        # If the closest neighbor has a distance of 0, then return it's output
        if dists[0][0] == 0:
            return target_buffer[neighbor_idx[0][0]]

        if self.data_window.size < self.n_neighbors:  # Select only the valid neighbors
            neighbor_vals = [
                target_buffer[index]
                for cnt, index in enumerate(neighbor_idx[0])
                if cnt < self.data_window.size
            ]
            dists = [dist for cnt, dist in enumerate(dists[0]) if cnt < self.data_window.size]
        else:
            neighbor_vals = [target_buffer[index] for index in neighbor_idx[0]]
            dists = dists[0]

        if self.aggregation_method == self._MEAN:
            return np.mean(neighbor_vals)
        elif self.aggregation_method == self._MEDIAN:
            return np.median(neighbor_vals)
        else:  # weighted mean
            return sum(y / d for y, d in zip(neighbor_vals, dists)) / sum(1 / d for d in dists)
