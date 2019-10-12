import collections
import operator

from .. import base
from .. import utils


__all__ = ['KNeighborsRegressor', 'KNeighborsClassifier']


class NearestNeighbours(collections.deque):

    def __init__(self, window_size, p):
        super().__init__(self, maxlen=window_size)
        self.p = p

    def update(self, x, y):
        super().append((x, y))
        return self

    def find_nearest(self, x, k):
        """Returns the ``k`` closest points, along with their distances."""

        # Compute the distances to each point in the window
        points = ((*p, utils.minkowski_distance(a=x, b=p[0], p=self.p)) for p in self)

        # Return the k closest points
        return sorted(points, key=operator.itemgetter(2))[:k]


class KNeighborsRegressor(NearestNeighbours, base.Regressor):
    """K-Nearest Neighbors (KNN) for regression.

    Parameters:
        n_neighbors (int): Number of neighbors to use.
        window_size (int): Size of the sliding window use to search neighbors with.
        p (int): Power parameter for the Minkowski metric. When ``p=1``, this corresponds to the
            Manhattan distance, while ``p=2`` corresponds to the Euclidean distance.
        weighted (bool): Whether to weight the contribution of each neighbor by it's inverse
            distance or not.

    Example:

        ::

            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import neighbors
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=True,
            ...     random_state=42
            ... )

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     neighbors.KNeighborsRegressor()
            ... )

            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 3.817195

    """

    def __init__(self, n_neighbors=5, window_size=50, p=2, weighted=True):
        super().__init__(window_size=window_size, p=p)
        self.n_neighbors = n_neighbors
        self.weighted = weighted

    def fit_one(self, x, y):
        return super().update(x, y)

    def predict_one(self, x):

        nearest = self.find_nearest(x=x, k=self.n_neighbors)

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


class KNeighborsClassifier(NearestNeighbours, base.MultiClassifier):
    """K-Nearest Neighbors (KNN) for classification.

    Parameters:
        n_neighbors (int): Number of neighbors to use.
        window_size (int): Size of the sliding window use to search neighbors with.
        p (int): Power parameter for the Minkowski metric. When ``p=1``, this corresponds to the
            Manhattan distance, while ``p=2`` corresponds to the Euclidean distance.
        weighted (bool): Whether to weight the contribution of each neighbor by it's inverse
            distance or not.

    Example:

        ::

            >>> from creme import datasets
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import neighbors
            >>> from creme import preprocessing

            >>> X_y = datasets.fetch_electricity()

            >>> model = (
            ...     preprocessing.StandardScaler() |
            ...     neighbors.KNeighborsClassifier()
            ... )

            >>> metric = metrics.Accuracy()

            >>> model_selection.online_score(X_y, model, metric)
            Accuracy: 0.88526

    """

    def __init__(self, n_neighbors=5, window_size=50, p=2, weighted=True):
        super().__init__(window_size=window_size, p=p)
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.classes = set()

    def fit_one(self, x, y):
        self.classes.add(y)
        return super().update(x, y)

    def predict_proba_one(self, x):

        nearest = self.find_nearest(x=x, k=self.n_neighbors)

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
        return utils.softmax(y_pred)
