from skmultiflow.lazy import KNNClassifier
from river.drift import ADWIN
from skmultiflow.utils.utils import get_dimensions

import numpy as np

import warnings


def KNNAdwin(n_neighbors=5, max_window_size=1000,
             leaf_size=30):     # pragma: no cover
    warnings.warn("'KNNAdwin' has been renamed to 'KNNADWINClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return KNNADWINClassifier(n_neighbors=n_neighbors,
                              max_window_size=max_window_size,
                              leaf_size=leaf_size)


class KNNADWINClassifier(KNNClassifier):
    """ K-Nearest Neighbors classifier with ADWIN change detector.

    This Classifier is an improvement from the regular KNNClassifier,
    as it is resistant to concept drift. It utilises the ADWIN change
    detector to decide which samples to keep and which ones to forget,
    and by doing so it regulates the sample window size.

    To know more about the ADWIN change detector, please see
    :class:`skmultiflow.drift.ADWIN`

    It uses the regular KNNClassifier as a base class, with the
    major difference that this class keeps a variable size window,
    instead of a fixed size one and also it updates the adwin algorithm
    at each partial_fit call.

    Parameters
    ----------
    n_neighbors: int (default=5)
        The number of nearest neighbors to search for.

    max_window_size: int (default=1000)
        The maximum size of the window storing the last viewed samples.

    leaf_size: int (default=30)
        The maximum number of samples that can be stored in one leaf node,
        which determines from which point the algorithm will switch for a
        brute-force approach. The bigger this number the faster the tree
        construction time, but the slower the query time will be.

    metric: string or sklearn.DistanceMetric object
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
    >>> from skmultiflow.lazy import KNNADWINClassifier
    >>> from skmultiflow.data import ConceptDriftStream
    >>> # Setting up the stream
    >>> stream = ConceptDriftStream(position=2500, width=100, random_state=1)
    >>> # Setting up the KNNAdwin classifier
    >>> knn_adwin = KNNADWINClassifier(n_neighbors=8, leaf_size=40, max_window_size=1000)
    >>> # Keep track of sample count and correct prediction count
    >>> n_samples = 0
    >>> corrects = 0
    >>> while n_samples < 5000:
    ...     X, y = stream.next_sample()
    ...     pred = knn_adwin.predict(X)
    ...     if y[0] == pred[0]:
    ...         corrects += 1
    ...     knn_adwin = knn_adwin.partial_fit(X, y)
    ...     n_samples += 1
    >>>
    >>> # Displaying the results
    >>> print('KNNClassifier usage example')
    >>> print(str(n_samples) + ' samples analyzed.')
    5000 samples analyzed.
    >>> print("KNNADWINClassifier's performance: " + str(corrects/n_samples))
    KNNAdwin's performance: 0.5714

    """

    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean'):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric)
        self.adwin = ADWIN()

    def reset(self):
        """ Reset the estimator.

        Resets the ADWIN Drift detector as well as the KNN model.

        Returns
        -------
        KNNADWINClassifier
            self

        """
        self.adwin = ADWIN()
        return super().reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all
            samples in X.

        classes: numpy.ndarray, optional (default=None)
            Array with all possible/known classes.

        sample_weight: Not used.

        Returns
        -------
        KNNADWINClassifier
            self

        Notes
        -----
        Partially fits the model by updating the window with new samples
        while also updating the ADWIN algorithm. IF ADWIN detects a change,
        the window is split in such a wat that samples from the previous
        concept are dropped.

        """
        r, c = get_dimensions(X)
        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        for i in range(r):
            self.data_window.add_sample(X[i], y[i])
            if self.data_window.size >= self.n_neighbors:
                correctly_classifies = 1 if self.predict(np.asarray([X[i]])) == y[i] else 0
                self.adwin.update(correctly_classifies)
            else:
                self.adwin.update(0)

        if self.data_window.size >= self.n_neighbors:
            if self.adwin.change_detected:
                if self.adwin.width < self.data_window.size:
                    for i in range(self.data_window.size, self.adwin.width, -1):
                        self.data_window.delete_oldest_sample()
        return self
