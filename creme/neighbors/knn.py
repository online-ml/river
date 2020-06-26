import collections
import operator

from creme import base
from creme import utils


__all__ = ['KNeighborsRegressor', 'KNeighborsClassifier']


class NearestNeighbours:

    def __init__(self, window_size, p):
        self.window_size = window_size
        self.p = p
        self.window = collections.deque(maxlen=window_size)

    def update(self, x, y):
        self.window.append((x, y))
        return self

    def find_nearest(self, x, k):
        """Returns the `k` closest points to `x`, along with their distances."""

        # Compute the distances to each point in the window
        points = ((*p, utils.math.minkowski_distance(a=x, b=p[0], p=self.p)) for p in self.window)

        # Return the k closest points
        return sorted(points, key=operator.itemgetter(2))[:k]


class KNeighborsRegressor(base.Regressor):
    """K-Nearest Neighbors (KNN) for regression.

    This works by storing a buffer with the `window_size` most recent observations. A brute-force
    search is used to find the `n_neighbors` nearest observations in the buffer to make a
    prediction.

    Parameters:
        n_neighbors: Number of neighbors to use.
        window_size: Size of the sliding window use to search neighbors with.
        p: Power parameter for the Minkowski metric. When `p=1`, this corresponds to the
            Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        weighted: Whether to weight the contribution of each neighbor by it's inverse distance or
            not.

    Example:

        >>> from creme import datasets
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import neighbors
        >>> from creme import preprocessing

        >>> X_y = datasets.TrumpApproval()

        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     neighbors.KNeighborsRegressor()
        ... )

        >>> metric = metrics.MAE()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        MAE: 0.335753

    """

    def __init__(self, n_neighbors=5, window_size=50, p=2, weighted=True):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.weighted = weighted
        self.p = p
        self._nn = NearestNeighbours(window_size=window_size, p=p)

    def fit_one(self, x, y):
        self._nn.update(x, y)
        return self

    def predict_one(self, x):

        nearest = self._nn.find_nearest(x=x, k=self.n_neighbors)

        if not nearest:
            return 0.

        # If the closest neighbor has a distance of 0, then return it's output
        if nearest[0][2] == 0:
            return nearest[0][1]

        # Weighted average
        if self.weighted:
            return (
                sum(y / d for _, y, d in nearest) /
                sum(1 / d for _, _, d in nearest)
            )

        # Uniform average
        return sum(y for _, y, _ in nearest) / self.n_neighbors


class KNeighborsClassifier(base.MultiClassifier):
    """K-Nearest Neighbors (KNN) for classification.

    This works by storing a buffer with the `window_size` most recent observations. A brute-force
    search is used to find the `n_neighbors` nearest observations in the buffer to make a
    prediction.

    Parameters:
        n_neighbors: Number of neighbors to use.
        window_size: Size of the sliding window use to search neighbors with.
        p: Power parameter for the Minkowski metric. When `p=1`, this corresponds to the
            Manhattan distance, while `p=2` corresponds to the Euclidean distance.
        weighted: Whether to weight the contribution of each neighbor by it's inverse
            distance or not.

    Example:

        >>> from creme import datasets
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import neighbors
        >>> from creme import preprocessing

        >>> X_y = datasets.Phishing()

        >>> model = (
        ...     preprocessing.StandardScaler() |
        ...     neighbors.KNeighborsClassifier()
        ... )

        >>> metric = metrics.Accuracy()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        Accuracy: 84.55%

    """

    def __init__(self, n_neighbors=5, window_size=50, p=2, weighted=True):
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.weighted = weighted
        self.p = p
        self.classes = set()
        self._nn = NearestNeighbours(window_size=window_size, p=p)

    def fit_one(self, x, y):
        self.classes.add(y)
        self._nn.update(x, y)
        return self

    def predict_proba_one(self, x):

        nearest = self._nn.find_nearest(x=x, k=self.n_neighbors)

        y_pred = {c: 0. for c in self.classes}

        if not nearest:
            return y_pred

        # If the closest neighbor has a distance of 0, then return it's output
        if nearest[0][2] == 0:
            y_pred[nearest[0][1]] = 1.
            return y_pred

        # Weighted votes
        if self.weighted:
            for _, y, d in nearest:
                y_pred[y] += 1. / d

        # Uniform votes
        else:
            for _, y, _ in nearest:
                y_pred[y] += 1.

        # Normalize votes into real [0, 1] probabilities
        return utils.math.softmax(y_pred)
