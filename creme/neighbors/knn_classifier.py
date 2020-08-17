import typing

from creme import base
from creme.utils import dict2numpy
from creme.utils.math import softmax

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
        scipy.spatial.cKDTree parameter. The maximum number of samples that
        can be stored in one leaf node, which determines from which point
        the algorithm will switch for a brute-force approach. The bigger
        this number the faster the tree construction time, but the slower
        the query time will be.

    p : float (default=2)
        p-norm value for the Minkowski metric. When `p=1`, this corresponds to the
        Manhattan distance, while `p=2` corresponds to the Euclidean distance. Valid
        values are in the interval $[1, +\infty)$

    weighted : bool (default=True)
        Whether to weight the contribution of each neighbor by it's inverse
        distance or not.

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features. This implementation treats all features from a given stream as
    numerical.

    Examples
    --------
        >>> from creme import datasets
        >>> from creme import evaluate
        >>> from creme import metrics
        >>> from creme import neighbors
        >>> from creme import preprocessing

        >>> dataset = datasets.Phishing()

        >>> model = (
        ...    preprocessing.StandardScaler() |
        ...    neighbors.KNNClassifier()
        ... )

        >>> metric = metrics.Accuracy()

        >>> evaluate.progressive_val_score(dataset, model, metric)
        Accuracy: 88.11%

    """

    def __init__(self, n_neighbors: int = 5, max_window_size: int = 1000, leaf_size: int = 30,
                 p: float = 2, weighted: bool = True):
        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         p=p)
        self.weighted = weighted
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

        dists, neighbor_idx = self._get_neighbors(x_arr)

        target_buffer = self.data_window.targets_buffer

        if not self.weighted:  # Uniform weights
            for index in neighbor_idx[0]:
                proba[target_buffer[index]] += 1.
        else:  # Use the inverse of the distance to weight the votes
            for d, index in zip(dists[0], neighbor_idx[0]):
                proba[target_buffer[index]] += 1. / d

        return softmax(proba)

    @property
    def _multiclass(self):
        return True
