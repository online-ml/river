import collections
import functools
import operator

from .. import base
from .. import dist
from .. import utils


__all__ = ['GaussianNB']


class GaussianNB(base.MultiClassifier):
    """Gaussian Naive Bayes.

    Example:

        >>> from creme import naive_bayes
        >>> from creme import stream
        >>> import numpy as np

        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> Y = np.array([1, 1, 1, 2, 2, 2])

        >>> model = naive_bayes.GaussianNB()

        >>> for x, y in stream.iter_numpy(X, Y):
        ...     _ = model.fit_one(x, y)

        >>> model.predict_one({0: -0.8, 1: -1})
        1

    """

    def __init__(self):
        dd = collections.defaultdict
        self.gaussians = dd(functools.partial(dd, dist.Normal))

    def fit_one(self, x, y):

        y_pred = self.predict_proba_one(x)

        for i, xi in x.items():
            self.gaussians[y][i].update(xi)

        return y_pred

    def predict_proba_one(self, x):
        return utils.normalize_y_pred({
            y: functools.reduce(
                operator.mul,
                (gaussians[i].pdf(xi) for i, xi in x.items()),
                1
            )
            for y, gaussians in self.gaussians.items()
        })
