import collections

from .. import base
from .. import stats


class VarianceThreshold(base.Transformer):
    """Removes low-variance features.

    Properties:
        threshold (float): Only features with a variance above the threshold will be kept.
        min_samples (int): The minimum number of samples required to perform selection.

    Attributes:
        variances (dict): The variance of each feature.

    Example:

        ::

            >>> from creme import feature_selection
            >>> from creme import stream

            >>> X = [
            ...     [0, 2, 0, 3],
            ...     [0, 1, 4, 3],
            ...     [0, 1, 1, 3]
            ... ]

            >>> selector = feature_selection.VarianceThreshold()

            >>> for x, _ in stream.iter_array(X):
            ...     print(selector.fit_one(x).transform_one(x))
            {0: 0, 1: 2, 2: 0, 3: 3}
            {1: 1, 2: 4}
            {1: 1, 2: 1}

    """

    def __init__(self, threshold=0.0, min_samples=2):
        self.threshold = threshold
        self.min_samples = min_samples
        self.variances = collections.defaultdict(stats.Var)

    def fit_one(self, x, y=None):

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
        return {
            i: xi
            for i, xi in x.items()
            if self.check_feature(i)
        }
