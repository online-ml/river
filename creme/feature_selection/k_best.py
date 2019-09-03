import collections
import copy
import functools

from .. import base


class SelectKBest(base.Transformer):
    """Removes all but the $k$ highest scoring features.

    Parameters:
        similarity (stats.Bivariate)
        k (int)

    Attributes:
        similarities (dict): The similarity instances used for each feature.
        leaderboard (dict): The actual similarity measures.

    Example:

        ::

            >>> from pprint import pprint
            >>> from creme import feature_selection
            >>> from creme import stats
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X, y = datasets.make_regression(
            ...     n_samples=100,
            ...     n_features=10,
            ...     n_informative=2,
            ...     random_state=42
            ... )

            >>> selector = feature_selection.SelectKBest(
            ...     similarity=stats.PearsonCorrelation(),
            ...     k=2
            ... )

            >>> for xi, yi, in stream.iter_array(X, y):
            ...     selector = selector.fit_one(xi, yi)

            >>> pprint(selector.leaderboard)
            Counter({9: 0.789828...,
                     7: 0.544461...,
                     8: 0.106205...,
                     2: 0.063886...,
                     4: 0.053821...,
                     5: 0.027151...,
                     1: -0.031251...,
                     6: -0.065716...,
                     3: -0.150104...,
                     0: -0.189527...})

            >>> selector.transform_one(xi)
            {7: -1.279576..., 9: -1.840874...}

    """

    def __init__(self, similarity, k=10):
        self.k = k
        self.similarities = collections.defaultdict(functools.partial(copy.deepcopy, similarity))
        self.leaderboard = collections.Counter()

    @property
    def is_supervised(self):
        return True

    def fit_one(self, x, y):

        for i, xi in x.items():
            self.leaderboard[i] = self.similarities[i].update(xi, y).get()

        return self

    def transform_one(self, x):

        best_features = set(pair[0] for pair in self.leaderboard.most_common(self.k))

        if self.leaderboard:

            return {
                i: xi
                for i, xi in x.items()
                if i in best_features
            }

        return copy.deepcopy(x)
