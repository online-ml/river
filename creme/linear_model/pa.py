import collections

import numpy as np

from .. import base
from .. import optim
from .. import utils


__all__ = ['PAClassifier', 'PARegressor']


class BasePA:

    def __init__(self, C, mode, fit_intercept):
        self.C = C
        self.calc_tau = {0: self._calc_tau_0, 1: self._calc_tau_1, 2: self._calc_tau_2}[mode]
        self.fit_intercept = fit_intercept
        self.weights = collections.defaultdict(float)
        self.intercept = 0.

    @classmethod
    def _calc_tau_0(cls, x, loss):
        norm = utils.norm(x, order=2) ** 2
        if norm > 0:
            return loss / utils.norm(x, order=2) ** 2
        return 0

    def _calc_tau_1(self, x, loss):
        norm = utils.norm(x, order=2) ** 2
        if norm > 0:
            return min(self.C, loss / norm)
        return 0

    def _calc_tau_2(self, x, loss):
        return loss / (utils.norm(x, order=2) ** 2 + 0.5 / self.C)


class PARegressor(BasePA, base.Regressor):
    """Passive-aggressive learning for regression.

    Example:

        The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import stream
            >>> import numpy as np
            >>> from sklearn import datasets

            >>> np.random.seed(1000)
            >>> X, y = datasets.make_regression(n_samples=500, n_features=4)

            >>> model = linear_model.PARegressor(
            ...     C=0.01,
            ...     mode=2,
            ...     eps=0.1,
            ...     fit_intercept=False
            ... )
            >>> metric = metrics.MAE() + metrics.MSE()

            >>> for xi, yi in stream.iter_array(X, y):
            ...     y_pred = model.predict_one(xi)
            ...     model = model.fit_one(xi, yi)
            ...     metric = metric.update(yi, y_pred)

            >>> print(metric)
            MAE: 10.123199, MSE: 843.816135

        References:
            1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def __init__(self, C=1.0, mode=1, eps=0.1, fit_intercept=True):
        super().__init__(C=C, mode=mode, fit_intercept=fit_intercept)
        self.loss = optim.losses.EpsilonInsensitiveHinge(eps=eps)

    def fit_one(self, x, y):

        y_pred = self.predict_one(x)
        tau = self.calc_tau(x, self.loss.eval(y, y_pred))
        step = tau * np.sign(y - y_pred)

        for i, xi in x.items():
            self.weights[i] += step * xi
        if self.fit_intercept:
            self.intercept += step

        return self

    def predict_one(self, x):
        return utils.dot(x, self.weights) + self.intercept


class PAClassifier(BasePA, base.BinaryClassifier):
    """Passive-aggressive learning for classification.

    Example:

        The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

        ::

            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import stream
            >>> import numpy as np
            >>> from sklearn import datasets
            >>> from sklearn import model_selection

            >>> np.random.seed(1000)
            >>> X, y = datasets.make_classification(
            ...     n_samples=5000,
            ...     n_features=4,
            ...     n_informative=2,
            ...     n_redundant=0,
            ...     n_repeated=0,
            ...     n_classes=2,
            ...     n_clusters_per_class=2
            ... )

            >>> X_train, X_test, y_train, y_test = model_selection.train_test_split(
            ...     X,
            ...     y,
            ...     test_size=0.35,
            ...     random_state=1000
            ... )

            >>> model = linear_model.PAClassifier(
            ...     C=0.01,
            ...     mode=1
            ... )

            >>> for xi, yi in stream.iter_array(X_train, y_train):
            ...     y_pred = model.fit_one(xi, yi)

            >>> metric = metrics.Accuracy() + metrics.LogLoss()

            >>> for xi, yi in stream.iter_array(X_test, y_test):
            ...     metric = metric.update(yi, model.predict_proba_one(xi))

            >>> print(metric)
            Accuracy: 0.884571, LogLoss: 0.325727

    References:
        1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def __init__(self, C=1.0, mode=1, fit_intercept=True):
        super().__init__(C=C, mode=mode, fit_intercept=fit_intercept)
        self.loss = optim.losses.Hinge()

    def fit_one(self, x, y):

        y_pred = utils.dot(x, self.weights) + self.intercept
        tau = self.calc_tau(x, self.loss.eval(y, y_pred))
        step = tau * (y or -1)  # y == False becomes -1

        for i, xi in x.items():
            self.weights[i] += step * xi
        if self.fit_intercept:
            self.intercept += step

        return self

    def predict_proba_one(self, x):
        y_pred = utils.sigmoid(utils.dot(x, self.weights) + self.intercept)
        return {False: 1. - y_pred, True: y_pred}
