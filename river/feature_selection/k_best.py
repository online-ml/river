from __future__ import annotations

import collections
import copy
import functools
import typing

from river import base, stats


class SelectKBest(base.SupervisedTransformer):
    """Removes all but the $k$ highest scoring features.

    Parameters
    ----------
    similarity
    k
        The number of features to keep.
    use_abs
        A boolean indicating whether to rank features based on the absolute value of
        their scores. This is particularly useful when the similarity metric can produce
        negative values, such as Pearson correlation. Defaults to `False`.

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
    ...     selector.learn_one(xi, yi)

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

    >>> # Example demonstrating the `use_abs` parameter

    >>> import numpy as np

    >>> np.random.seed(42)
    >>> X_abs = np.random.rand(100, 3)
    >>> y_abs = (
    ...     0.6 * X_abs[:, 0]
    ...     - 0.9 * X_abs[:, 1]
    ...     + 0.1 * X_abs[:, 2]
    ...     + np.random.normal(0, 0.1, 100)
    ... )

    >>> # With use_abs=False (the default), feature 0 is selected.
    >>> selector_no_abs = feature_selection.SelectKBest(
    ...     stats.PearsonCorr(),
    ...     k=1,
    ...     use_abs=False
    ... )

    >>> for xi, yi in stream.iter_array(X_abs, y_abs):
    ...     selector_no_abs.learn_one(xi, yi)

    >>> pprint(selector_no_abs.leaderboard)
    Counter({0: 0.46719795106878087,
            2: 0.09739583197225814,
            1: -0.8018812540692073})

    >>> selector_no_abs.transform_one({i: v for i, v in enumerate(X_abs[-1])})
    {0: np.float64(0.21582102749684318)}

    >>> # With use_abs=True, feature 1 is selected due to its larger absolute correlation.
    >>> selector_with_abs = feature_selection.SelectKBest(stats.PearsonCorr(), k=1, use_abs=True)

    >>> for xi, yi in stream.iter_array(X_abs, y_abs):
    ...     selector_with_abs.learn_one(xi, yi)

    >>> selector_with_abs.transform_one({i: v for i, v in enumerate(X_abs[-1])})
    {1: np.float64(0.6228904758190003)}
    """

    def __init__(self, similarity: stats.base.Bivariate, use_abs: bool = False, k=10):
        self.k = k
        self.similarity = similarity
        self.similarities: collections.defaultdict = collections.defaultdict(
            functools.partial(copy.deepcopy, similarity)
        )
        self.leaderboard: typing.Counter = collections.Counter()
        self.use_abs = use_abs

    @classmethod
    def _unit_test_params(cls):
        yield {"similarity": stats.PearsonCorr()}

    def learn_one(self, x, y):
        for i, xi in x.items():
            self.similarities[i].update(xi, y)
            if self.use_abs:
                similarity_value = abs(self.similarities[i].get())
            else:
                similarity_value = self.similarities[i].get()
            self.leaderboard[i] = similarity_value

    def transform_one(self, x):
        best_features = {pair[0] for pair in self.leaderboard.most_common(self.k)}

        if self.leaderboard:
            return {i: xi for i, xi in x.items() if i in best_features}

        return copy.deepcopy(x)
