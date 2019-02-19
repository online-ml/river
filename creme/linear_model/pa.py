import abc
import collections

import numpy as np

from .. import base
from .. import optim

from . import util


class PassiveAggressiveMixin:

    def _calculate_tau(self, x, loss):
        return loss / (np.power(np.linalg.norm(list(x.values()), ord=2), 2) + 10e-10)


class PassiveAggressiveIMixin:

    def _calculate_tau(self, x, loss):
        return min(self.C, loss / (np.power(np.linalg.norm(list(x.values()), ord=2), 2) + 10e-10))


class PassiveAggressiveIIMixin:

    def _calculate_tau(self, x, loss):
        return loss / (np.power(np.linalg.norm(list(x.values()), ord=2), 2) + (1 / (2 * self.C)) + 10e-10)


class BasePassiveAggressiveClassifier(base.BinaryClassifier, abc.ABC):

    def __init__(self, C=0.01):
        self.C = C
        self.loss = optim.HingeLoss()
        self.weights = collections.defaultdict(float)

    @abc.abstractmethod
    def _calculate_tau(self, x, loss):
        pass

    def fit_one(self, x, y):

        # PA algorithms assume y is in {-1, 1}
        y = y or -1

        y_pred = util.dot(x, self.weights)
        loss = self.loss(y, y_pred)
        tau = self._calculate_tau(x, loss)
        step = tau * y

        for i, xi in x.items():
            self.weights[i] += step * xi

        return util.clip(y_pred)

    def predict_one(self, x):
        return util.dot(x, self.weights) > 0

    def predict_proba_one(self, x):
        return util.clip(util.dot(x, self.weights))


class PassiveAggressiveClassifier(PassiveAggressiveMixin, BasePassiveAggressiveClassifier):
    """Passive-aggressive classifier that assumes classes are separable.

    Example:

    The following example is taken from `here <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

    ::

        >>> from creme import linear_model
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

        >>> model = linear_model.PassiveAggressiveClassifier(C=0.01)

        >>> for xi, yi in stream.iter_numpy(X_train, y_train):
        ...     y_pred = model.fit_one(xi, yi)

        >>> n_correct = 0

        >>> for xi, yi in stream.iter_numpy(X_test, y_test):
        ...     n_correct += model.predict_one(xi) == yi

        >>> n_correct / len(X_test)
        0.749142...

    References:
        - `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """
    pass


class PassiveAggressiveIClassifier(PassiveAggressiveIMixin, BasePassiveAggressiveClassifier):
    """Variant I of the passive-aggressive classifier.

    Example:

    The following example is taken from `here <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

    ::

        >>> from creme import linear_model
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

        >>> model = linear_model.PassiveAggressiveIClassifier(C=0.01)

        >>> for xi, yi in stream.iter_numpy(X_train, y_train):
        ...     y_pred = model.fit_one(xi, yi)

        >>> n_correct = 0

        >>> for xi, yi in stream.iter_numpy(X_test, y_test):
        ...     n_correct += model.predict_one(xi) == yi

        >>> n_correct / len(X_test)
        0.884571...

    References:
        - `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """
    pass


class PassiveAggressiveIIClassifier(PassiveAggressiveIIMixin, BasePassiveAggressiveClassifier):
    """Variant II of the passive-aggressive classifier.

    Example:

    The following example is taken from `here <https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/>`_.

    ::

        >>> from creme import linear_model
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

        >>> model = linear_model.PassiveAggressiveIIClassifier(C=0.01)

        >>> for xi, yi in stream.iter_numpy(X_train, y_train):
        ...     y_pred = model.fit_one(xi, yi)

        >>> n_correct = 0

        >>> for xi, yi in stream.iter_numpy(X_test, y_test):
        ...     n_correct += model.predict_one(xi) == yi

        >>> n_correct / len(X_test)
        0.884

    References:
        - `Online Passive-Aggressive Algorithms <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_

    """
    pass


class BasePassiveAggressiveRegressor(base.Regressor, abc.ABC):

    def __init__(self, C=0.01, eps=0.1):
        self.C = C
        self.loss = optim.EpsilonInsensitiveHingeLoss(eps=0.1)
        self.weights = collections.defaultdict(float)

    @abc.abstractmethod
    def _calculate_tau(self, x, loss):
        pass

    def fit_one(self, x, y):

        y_pred = util.dot(x, self.weights)
        loss = self.loss(y, y_pred)
        tau = self._calculate_tau(x, loss)
        step = tau * np.sign(y - y_pred)

        for i, xi in x.items():
            self.weights[i] += step * xi

        return y_pred

    def predict_one(self, x):
        return util.dot(x, self.weights)


class PassiveAggressiveRegressor(PassiveAggressiveMixin, BasePassiveAggressiveRegressor):
    """Passive-aggressive regressor.

    Example:

    ::

        >>> from creme import linear_model
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X, y = datasets.make_regression(n_features=4, random_state=0)
        >>> model = linear_model.PassiveAggressiveRegressor(C=1., eps=.1)

        >>> for xi, yi in stream.iter_numpy(X, y):
        ...     y_pred = model.fit_one(xi, yi)

        >>> list(model.weights.values())
        [20.451551..., 34.173492..., 67.618053..., 87.932659...]

    """
    pass


class PassiveAggressiveIRegressor(PassiveAggressiveIMixin, BasePassiveAggressiveRegressor):
    """Variant I of the passive-aggressive regressor.

    Example:

    ::

        >>> from creme import linear_model
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X, y = datasets.make_regression(n_features=4, random_state=0)
        >>> model = linear_model.PassiveAggressiveIRegressor(C=1., eps=.1)

        >>> for xi, yi in stream.iter_numpy(X, y):
        ...     y_pred = model.fit_one(xi, yi)

        >>> list(model.weights.values())
        [18.525988..., 29.719902..., 42.225340..., 59.661269...]

    """
    pass


class PassiveAggressiveIIRegressor(PassiveAggressiveIIMixin, BasePassiveAggressiveRegressor):
    """Variant II of the passive-aggressive regressor.

    Example:

    ::

        >>> from creme import linear_model
        >>> from creme import stream
        >>> from sklearn import datasets

        >>> X, y = datasets.make_regression(n_features=4, random_state=0)
        >>> model = linear_model.PassiveAggressiveIIRegressor(C=1., eps=.1)

        >>> for xi, yi in stream.iter_numpy(X, y):
        ...     y_pred = model.fit_one(xi, yi)

        >>> list(model.weights.values())
        [20.505129..., 34.210888..., 67.612151..., 87.897829...]

    """
    pass
