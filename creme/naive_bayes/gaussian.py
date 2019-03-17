import collections
import functools
import math

from .. import dist

from . import base


__all__ = ['GaussianNB']


class GaussianNB(base.BaseNB):
    """Gaussian Naive Bayes.

    This class inherits ``predict_proba_one`` from ``naive_bayes.BaseNB`` which itself inherits
    ``predict_one`` from ``base.MultiClassifier``.

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
        self.class_dist = dist.Multinomial()
        defaultdict = collections.defaultdict
        self.gaussians = defaultdict(functools.partial(defaultdict, dist.Normal))

    def fit_one(self, x, y):

        self.class_dist.update(y)

        for i, xi in x.items():
            self.gaussians[y][i].update(xi)

        return self

    def _joint_log_likelihood(self, x):
        return {
            c: math.log(self.class_dist.pmf(c)) + sum(
                math.log(10e-10 + gaussians[i].pdf(xi))
                for i, xi in x.items()
            )
            for c, gaussians in self.gaussians.items()
        }
