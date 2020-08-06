import numpy as np

from creme import base
from creme.utils import dict2numpy

from .base_neighbors import BaseNeighbors


class KNNRegressor(BaseNeighbors, base.Regressor):
    """k-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last
    `max_window_size` training samples. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored-samples with
    respect to a query sample.

    Parameters
    ----------
    n_neighbors : int (default=5)
        The number of nearest neighbors to search for.

    max_window_size : int (default=1000)
        The maximum size of the window storing the last observed samples.

    leaf_size : int (default=30)
        sklearn.KDTree parameter. The maximum number of samples that can
        be stored in one leaf node, which determines from which point the
        algorithm will switch for a brute-force approach. The bigger this
        number the faster the tree construction time, but the slower the
        query time will be.

    metric : string or sklearn.DistanceMetric object
        sklearn.KDTree parameter. The distance metric to use for the KDTree.
        Default=’euclidean’. KNNRegressor.valid_metrics() gives a list of
        the metrics which are valid for KDTree.

    aggregation_method : str (default='mean')
            | The method to aggregate the target values of neighbors.
            | 'mean'
            | 'median'

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data import RegressionGenerator
    >>> from skmultiflow.lazy import KNNRegressor
    >>> import numpy as np
    >>>
    >>> # Setup the data stream
    >>> stream = RegressionGenerator(random_state=1)
    >>> # Setup the estimator
    >>> knn = KNNRegressor()
    >>>
    >>> # Auxiliary variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 2000
    >>> y_pred = np.zeros(max_samples)
    >>> y_true = np.zeros(max_samples)
    >>>
    >>> # Run test-then-train loop for max_samples or while there is data in the stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    ...     X, y = stream.next_sample()
    ...     y_true[n_samples] = y[0]
    ...     y_pred[n_samples] = knn.predict(X)[0]
    ...     knn.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    2000 samples analyzed
    >>> print('KNN regressor mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))
    KNN regressor mean absolute error: 144.5672450178514

    """

    _MEAN = 'mean'
    _MEDIAN = 'median'

    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean',
                 aggregation_method='mean'):

        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric)
        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError('Invalid aggregation_method: {}.\n'
                             'Valid options are: {}'.format(aggregation_method,
                                                            {self._MEAN, self._MEDIAN}))
        self.aggregation_method = aggregation_method

    def fit_one(self, x: dict, y: base.typing.RegTarget) -> 'Regressor':
        """Fits to a set of features ``x`` and a real-valued target ``y``.

        Parameters
        ----------
            x: A dictionary of features.
            y: A numeric target.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the `max_window_size` is reached, removing older results.

        """

        x_arr = dict2numpy(x)
        self.data_window.add_one(X=x_arr, y=y)

        return self

    def predict_one(self, x: dict) -> base.typing.RegTarget:
        """Predicts the target value of a set of features `x`.

        Search the KDTree for the `n_neighbors` nearest neighbors.

        Parameters
        ----------
            x : A dictionary of features.

        Returns
        -------
            The prediction.

        """

        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # Not enough information available, return default prediction
            return None

        x_arr = dict2numpy(x)
        _, neighbors_idx = self._get_neighbors(x_arr)
        neighbors_val = [self.data_window.targets_buffer[idx] for idx in neighbors_idx]

        if self.aggregation_method == self._MEAN:
            y_pred = np.mean(neighbors_val)
        else:   # self.aggregation_method == self._MEDIAN
            y_pred = np.median(neighbors_val)

        return y_pred
