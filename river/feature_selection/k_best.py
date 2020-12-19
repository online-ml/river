import collections
import copy
import functools

from river import base
from river import stats


class SelectKBest(base.SupervisedTransformer):
    """Removes all but the $k$ highest scoring features.

    Parameters
    ----------
    similarity
    k
        The number of features to keep.

    Attributes
    ----------
    similarities : dict
        The similarity instances used for each feature.
    leaderboard : dict
        The actual similarity measures.

    Examples
    --------

    >>> from pprint import pprint
    >>> from river import feature_selection
    >>> from river import stats
    >>> from river import stream
    >>> from sklearn import datasets

    >>> X, y = datasets.make_regression(
    ...     n_samples=100,
    ...     n_features=10,
    ...     n_informative=2,
    ...     random_state=42
    ... )

    >>> selector = feature_selection.SelectKBest(
    ...     similarity=stats.PearsonCorr(),
    ...     k=2
    ... )

    >>> for xi, yi, in stream.iter_array(X, y):
    ...     selector = selector.learn_one(xi, yi)

    >>> pprint(selector.leaderboard)
    Counter({9: 0.7898,
            7: 0.5444,
            8: 0.1062,
            2: 0.0638,
            4: 0.0538,
            5: 0.0271,
            1: -0.0312,
            6: -0.0657,
            3: -0.1501,
            0: -0.1895})

    >>> selector.transform_one(xi)
    {7: -1.2795, 9: -1.8408}

    """

    def __init__(self, similarity: stats.Bivariate, k=10):
        self.k = k
        self.similarity = similarity
        self.similarities = collections.defaultdict(functools.partial(copy.deepcopy, similarity))
        self.leaderboard = collections.Counter()

    @classmethod
    def _unit_test_params(cls):
        return {"similarity": stats.PearsonCorr()}

    def learn_one(self, x, y):

        for i, xi in x.items():
            self.leaderboard[i] = self.similarities[i].update(xi, y).get()

        return self

    def transform_one(self, x):

        best_features = set(pair[0] for pair in self.leaderboard.most_common(self.k))

        if self.leaderboard:

            return {i: xi for i, xi in x.items() if i in best_features}

        return copy.deepcopy(x)
