import numpy as np

from .. import base


class RandomDiscarder(base.Transformer):
    """Transformer that randomly discards features.

    Example:

    ::

        >>> from creme import feature_selection
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X, _ = datasets.make_regression(n_samples=30, n_features=8)
        >>> discarder = feature_selection.RandomDiscarder(n_to_keep=5)

        >>> for x, _ in stream.iter_numpy(X):
        ...     x_pruned = discarder.fit_one(x)
        ...     assert len(x_pruned) == 5
        ...     # creme's transformers are pure so the input should not have changed
        ...     assert len(x) == 8

    """

    def __init__(self, n_to_keep=None):
        self.n_to_keep = n_to_keep

    def fit_one(self, x, y=None):
        return self.transform_one(x)

    def transform_one(self, x):

        if len(x) - self.n_to_keep > 0:
            return {
                i: x[i]
                for i in np.random.choice(list(x.keys()), size=self.n_to_keep, replace=False)
            }

        return x
