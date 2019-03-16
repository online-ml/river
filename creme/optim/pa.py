import abc

import numpy as np

from .. import utils
from ..optim import losses

from . import base


__all__ = ['PassiveAggressiveI', 'PassiveAggressiveII']


class BasePassiveAggressive(base.Optimizer):

    def __init__(self, C=0.01):
        self.C = C

    @abc.abstractmethod
    def _calculate_tau(self, x, loss):
        """Calculates the amount by which to move ."""

    def update_weights(self, x, y, w, loss, f_pred, f_grad):

        y_pred = utils.dot(x, w)
        tau = self._calculate_tau(x, loss(y, y_pred))

        if isinstance(loss, losses.RegressionLoss):
            step = tau * np.sign(y - y_pred)
        else:
            step = tau * (y or -1)

        for i, xi in x.items():
            w[i] += step * xi

        return w, y_pred


class PassiveAggressiveI(BasePassiveAggressive):
    """Variant I of the passive-aggressive weight update algorithm.

    Example:

    The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

    ::

        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import optim
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

        >>> optimizer = optim.PassiveAggressiveI(C=0.01)
        >>> model = linear_model.LogisticRegression(
        ...     optimizer=optimizer,
        ...     loss=optim.HingeLoss()
        ... )

        >>> for xi, yi in stream.iter_numpy(X_train, y_train):
        ...     y_pred = model.fit_one(xi, yi)

        >>> metric = metrics.Accuracy()
        >>> for xi, yi in stream.iter_numpy(X_test, y_test):
        ...     metric = metric.update(yi, model.predict_one(xi))

        >>> print(metric)
        Accuracy: 0.884571

    References:

    1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def _calculate_tau(self, x, loss):
        norm = utils.norm(x, order=2) ** 2
        if norm > 0:
            return min(self.C, loss / norm)
        return 0


class PassiveAggressiveII(BasePassiveAggressive):
    """Variant II of the passive-aggressive weight update algorithm.

    Example:

    The following example is taken from `this blog post <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

    ::

        >>> from creme import linear_model
        >>> from creme import metrics
        >>> from creme import optim
        >>> from creme import stream
        >>> import numpy as np
        >>> from sklearn import datasets

        >>> np.random.seed(1000)
        >>> X, y = datasets.make_regression(n_samples=500, n_features=4)

        >>> optimizer = optim.PassiveAggressiveII(C=0.01)
        >>> model = linear_model.LinearRegression(
        ...     optimizer=optimizer,
        ...     loss=optim.EpsilonInsensitiveHingeLoss(eps=0.1),
        ...     intercept=False
        ... )
        >>> metric = metrics.MAE()

        >>> for xi, yi in stream.iter_numpy(X, y):
        ...     y_pred = model.predict_one(xi)
        ...     model = model.fit_one(xi, yi)
        ...     metric = metric.update(yi, y_pred)

        >>> print(metric)
        MAE: 10.123199

    References:

    1. `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """

    def _calculate_tau(self, x, loss):
        return loss / (utils.norm(x, order=2) ** 2 + 0.5 / self.C)
