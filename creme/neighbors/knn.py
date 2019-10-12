import collections
import operator

from .. import base


__all__ = ['KNeighborsRegressor']


def minkowski_distance(a, b, p):
    return sum((abs(a.get(k, 0.) - b.get(k, 0.))) ** p for k in set([*a.keys(), *b.keys()]))


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
        points = ((*p, minkowski_distance(a=x, b=p[0], p=self.p)) for p in self)

        # Return the k closest points
        return sorted(points, key=operator.itemgetter(2))[:k]


class KNeighborsRegressor(NearestNeighbours, base.Regressor):
    """K-Nearest Neighbors for regression.

    Parameters:
        n_neighbors (int): Number of neighbors to use.
        window_size (int): Size of the sliding window use to search neighbors with.
        p (int): Power parameter for the Minkowski metric. When ``p=1``, this corresponds to the
            Manhattan distance, while ``p=2`` corresponds to the Euclidean distance.
        weighted (bool): Whether to weight the contribution of each neighbor by it's inverse
            distance or not.

    Example:

        ::

            >>> from creme import compose
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

        # Weighted average
        if self.weighted:
            return (
                sum(y / d for _, y, d in nearest) /
                sum(1 / d for _, _, d in nearest)
            )

        # Uniform average
        return sum(y for _, y, _ in nearest) / self.n_neighbors
