from .knn_classifier import KNNClassifier
from river.drift import ADWIN
from river.utils import dict2numpy


class KNNADWINClassifier(KNNClassifier):
    """K-Nearest Neighbors classifier with ADWIN change detector.

    This classifier is an improvement from the regular kNN method,
    as it is resistant to concept drift. It uses the `ADWIN` change
    detector to decide which samples to keep and which ones to forget,
    and by doing so it regulates the sample window size.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last viewed samples.
    leaf_size
        The maximum number of samples that can be stored in one leaf node,
        which determines from which point the algorithm will switch for a
        brute-force approach. The bigger this number the faster the tree
        construction time, but the slower the query time will be.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance. Valid
        values are in the interval $[1, +\\infty)$

    Notes
    -----
    - This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.
    - This implementation is extended from the KNNClassifier, with the main
    difference that it keeps a dynamic window whose size changes in agreement
    with the amount of change detected by the ADWIN drift detector.

    Examples
    --------
    >>> from river import synth
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import neighbors

    >>> dataset = synth.ConceptDriftStream(position=500, width=20, seed=1).take(1000)

    >>> model = neighbors.KNNADWINClassifier(window_size=100)

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 57.36%

    """

    def __init__(self, n_neighbors=5, window_size=1000, leaf_size=30, p=2):
        super().__init__(n_neighbors=n_neighbors, window_size=window_size, leaf_size=leaf_size, p=p)
        self.adwin = ADWIN()

    def _unit_test_skips(self):
        return {"check_emerging_features", "check_disappearing_features"}

    def learn_one(self, x, y):
        """Update the model with a set of features `x` and a label `y`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The class label.

        Returns
        -------
            self

        Notes
        -----
        For the K-Nearest Neighbors Classifier, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results.

        """
        self.classes_.add(y)

        self.data_window.append(dict2numpy(x), y)
        if self.data_window.size >= self.n_neighbors:
            correctly_classifies = int(self.predict_one(x) == y)
            self.adwin.update(correctly_classifies)
        else:
            self.adwin.update(0)

        if self.data_window.size >= self.n_neighbors:
            if self.adwin.change_detected:
                if self.adwin.width < self.data_window.size:
                    for i in range(self.data_window.size, self.adwin.width, -1):
                        self.data_window.popleft()
        return self
