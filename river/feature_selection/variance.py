import collections

from river import base
from river import stats


class VarianceThreshold(base.Transformer):
    """Removes low-variance features.

    Parameters
    ----------
    threshold
        Only features with a variance above the threshold will be kept.
    min_samples
        The minimum number of samples required to perform selection.

    Attributes
    ----------
    variances : dict
        The variance of each feature.

    Examples
    --------

    >>> from river import feature_selection
    >>> from river import stream

    >>> X = [
    ...     [0, 2, 0, 3],
    ...     [0, 1, 4, 3],
    ...     [0, 1, 1, 3]
    ... ]

    >>> selector = feature_selection.VarianceThreshold()

    >>> for x, _ in stream.iter_array(X):
    ...     print(selector.learn_one(x).transform_one(x))
    {0: 0, 1: 2, 2: 0, 3: 3}
    {1: 1, 2: 4}
    {1: 1, 2: 1}

    """

    def __init__(self, threshold=0, min_samples=2):
        self.threshold = threshold
        self.min_samples = min_samples
        self.variances = collections.defaultdict(stats.Var)

    def learn_one(self, x):

        for i, xi in x.items():
            self.variances[i].update(xi)

        return self

    def check_feature(self, feature):
        if feature not in self.variances:
            return True
        if self.variances[feature].mean.n < self.min_samples:
            return True
        return self.variances[feature].get() > self.threshold

    def transform_one(self, x):
        return {i: xi for i, xi in x.items() if self.check_feature(i)}
