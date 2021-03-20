import typing

from river import base
from river.utils import dict2numpy
from river.utils.math import softmax

from .base_neighbors import BaseNeighbors


class KNNClassifier(BaseNeighbors, base.Classifier):
    """k-Nearest Neighbors classifier.

    This non-parametric classification method keeps track of the last
    `window_size` training samples. The predicted class-label for a
    given query sample is obtained in two steps:

    1. Find the closest `n_neighbors` to the query sample in the data window.
    2. Aggregate the class-labels of the `n_neighbors` to define the predicted
       class for the query sample.

    Parameters
    ----------
    n_neighbors
        The number of nearest neighbors to search for.
    window_size
        The maximum size of the window storing the last observed samples.
    leaf_size
        scipy.spatial.cKDTree parameter. The maximum number of samples that can be
        stored in one leaf node, which determines from which point the algorithm will
        switch for a brute-force approach. The bigger this number the faster the
        tree construction time, but the slower the query time will be.
    p
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance. Valid
        values are in the interval $[1, +\\infty)$
    weighted
        Whether to weight the contribution of each neighbor by it's inverse
        distance or not.
    kwargs
        Other parameters passed to `scipy.spatial.cKDTree`.

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

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     neighbors.KNNClassifier()
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 88.07%

    """

    def __init__(
        self,
        n_neighbors: int = 5,
        window_size: int = 1000,
        leaf_size: int = 30,
        p: float = 2,
        weighted: bool = True,
        **kwargs
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            leaf_size=leaf_size,
            p=p,
            **kwargs
        )
        self.weighted = weighted
        self.classes_: typing.Set = set()
        self.kwargs = kwargs

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
        x_arr = dict2numpy(x)

        self.data_window.append(x_arr, y)

        return self

    def predict_proba_one(self, x):
        """Predict the probability of each label for a dictionary of features `x`.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        proba
            A dictionary which associates a probability which each label.

        """

        proba = {class_idx: 0.0 for class_idx in self.classes_}
        if self.data_window.size == 0:
            # The model is empty, default to None
            return proba

        x_arr = dict2numpy(x)

        dists, neighbor_idx = self._get_neighbors(x_arr)
        target_buffer = self.data_window.targets_buffer

        # If the closest neighbor has a distance of 0, then return it's output
        if dists[0][0] == 0:
            proba[target_buffer[neighbor_idx[0][0]]] = 1.0
            return proba

        if self.data_window.size < self.n_neighbors:  # Select only the valid neighbors
            neighbor_idx = [
                index
                for cnt, index in enumerate(neighbor_idx[0])
                if cnt < self.data_window.size
            ]
            dists = [
                dist for cnt, dist in enumerate(dists[0]) if cnt < self.data_window.size
            ]
        else:
            neighbor_idx = neighbor_idx[0]
            dists = dists[0]

        if not self.weighted:  # Uniform weights
            for index in neighbor_idx:
                proba[target_buffer[index]] += 1.0
        else:  # Use the inverse of the distance to weight the votes
            for d, index in zip(dists, neighbor_idx):
                proba[target_buffer[index]] += 1.0 / d

        return softmax(proba)

    @property
    def _multiclass(self):
        return True
