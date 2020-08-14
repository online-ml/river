import typing

from creme import base
from creme.utils import dict2numpy

from .base_neighbors import BaseNeighbors


class KNNClassifier(BaseNeighbors, base.Classifier):
    """ k-Nearest Neighbors classifier.

    This non-parametric classification method keeps track of the last
    `max_window_size` training samples. The predicted class-label for a
    given query sample is obtained in two steps:

    1. Find the closest `n_neighbors` to the query sample in the data window.
    2. Aggregate the class-labels of the `n_neighbors` to define the predicted
       class for the query sample.

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
        Default=’euclidean’. KNNClassifier.valid_metrics() gives a list of
        the metrics which are valid for KDTree.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.lazy import KNNClassifier
    >>> from skmultiflow.data import SEAGenerator
    >>> # Setting up the stream
    >>> stream = SEAGenerator(random_state=1, noise_percentage=.1)
    >>> knn = KNNClassifier(n_neighbors=8, max_window_size=2000, leaf_size=40)
    >>> # Keep track of sample count and correct prediction count
    >>> n_samples = 0
    >>> corrects = 0
    >>> while n_samples < 5000:
    ...     X, y = stream.next_sample()
    ...     my_pred = knn.predict(X)
    ...     if y[0] == my_pred[0]:
    ...         corrects += 1
    ...     knn = knn.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Displaying results
    >>> print('KNNClassifier usage example')
    >>> print('{} samples analyzed.'.format(n_samples))
    5000 samples analyzed.
    >>> print("KNNClassifier's performance: {}".format(corrects/n_samples))
    KNN's performance: 0.8776

    """

    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean'):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric)
        self.classes = set()

    def learn_one(self, x: dict, y: base.typing.ClfTarget) -> 'Classifier':
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
            x : A dictionary of features.
            y : The class label.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors Classifier, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results.

        """

        self.classes.add(y)
        x_arr = dict2numpy(x)

        self.data_window.add_one(x_arr, y)

        return self

    def predict_proba_one(self, x: dict) -> typing.Dict[base.typing.ClfTarget, float]:
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters:
            x: A dictionary of features.

        Returns:
            A dictionary which associates a probability which each label.

        """

        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # The model is empty, default to None
            return None
        proba = {class_idx: 0.0 for class_idx in self.classes}
        x_arr = dict2numpy(x)

        _, neighbor_idx = self._get_neighbors(x_arr)
        for index in neighbor_idx:
            proba[self.data_window.targets_buffer[index]] += 1. / len(neighbor_idx)

        return proba

    @property
    def _multiclass(self):
        return True
