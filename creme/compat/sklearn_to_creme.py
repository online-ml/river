import copy
import functools
import typing

import numpy as np
from sklearn import base as sklearn_base
from sklearn import exceptions

from creme import base


__all__ = [
    'convert_sklearn_to_creme',
    'SKL2CremeClassifier',
    'SKL2CremeRegressor'
]


def convert_sklearn_to_creme(estimator: sklearn_base.BaseEstimator, batch_size=1,
                             classes: typing.Optional[list] = None):
    """Wraps a scikit-learn estimator to make it compatible with creme.

    Parameters:
        estimator
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only be updated once enough observations are available.
        classes: Class names necessary for classifiers.

    """

    if not hasattr(estimator, 'partial_fit'):
        raise ValueError(f'{estimator} does not have a partial_fit method')

    if isinstance(estimator, sklearn_base.ClassifierMixin) and classes is None:
        raise ValueError('classes must be provided to convert a classifier')

    wrappers = [
        (sklearn_base.RegressorMixin, functools.partial(
            SKL2CremeRegressor,
            batch_size=batch_size
        )),
        (sklearn_base.ClassifierMixin, functools.partial(
            SKL2CremeClassifier,
            batch_size=batch_size,
            classes=classes
        ))
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            return wrapper(copy.deepcopy(estimator))

    raise ValueError("Couldn't find an appropriate wrapper")


class SKL2CremeBase:

    def __init__(self, estimator: sklearn_base.BaseEstimator, batch_size: int,
                 x_dtype: typing.Type):
        self.estimator = estimator
        self.batch_size = batch_size
        self.x_dtype = x_dtype

        self._x_batch = None
        self._y_batch = None
        self._batch_i = 0


class SKL2CremeRegressor(SKL2CremeBase, base.Regressor):
    """`sklearn` to `creme` regressor adapter.

    Parameters:
        estimator: A scikit-learn transformer which has a `partial_fit` method.
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only update itself once enough observations are available.

    Example:

        >>> from creme import compat
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import linear_model
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     dataset=datasets.load_boston(),
        ...     shuffle=True,
        ...     seed=42
        ... )

        >>> scaler = preprocessing.StandardScaler()
        >>> sgd_reg = compat.convert_sklearn_to_creme(linear_model.SGDRegressor())
        >>> model = scaler | sgd_reg

        >>> metric = metrics.MAE()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        MAE: 11.004415

    """

    def __init__(self, estimator: sklearn_base.RegressorMixin, batch_size=1):
        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
            x_dtype=np.float
        )

    def fit_one(self, x, y):

        if self._x_batch is None:
            n_features = len(x)
            self._x_batch = np.empty(shape=(self.batch_size, n_features), dtype=self.x_dtype)
            self._y_batch = [None] * self.batch_size

        self._x_batch[self._batch_i, :] = list(x.values())
        self._y_batch[self._batch_i] = y

        self._batch_i += 1

        if self._batch_i == self.batch_size:
            self.estimator.partial_fit(X=self._x_batch, y=self._y_batch)
            self._batch_i = 0

        return self

    def predict_one(self, x):
        try:
            return self.estimator.predict(X=[list(x.values())])[0]
        except exceptions.NotFittedError:
            return 0


class SKL2CremeClassifier(SKL2CremeBase, base.MultiClassifier):
    """`sklearn` to `creme` classifier adapter.

    Parameters:
        estimator: A scikit-learn regressor which has a `partial_fit` method.
        classes
        batch_size: The amount of observations that will be used during each `partial_fit` call.
            Setting this to 1 means that the model will learn with each given observation.
            Increasing the batch size means that the observations will be stored in a buffer and
            the model will only update itself once enough observations are available.

    Example:

        >>> from creme import compat
        >>> from creme import metrics
        >>> from creme import model_selection
        >>> from creme import preprocessing
        >>> from creme import stream
        >>> from sklearn import linear_model
        >>> from sklearn import datasets

        >>> X_y = stream.iter_sklearn_dataset(
        ...     dataset=datasets.load_breast_cancer(),
        ...     shuffle=True,
        ...     seed=42
        ... )

        >>> model = preprocessing.StandardScaler()
        >>> model |= compat.convert_sklearn_to_creme(
        ...     estimator=linear_model.SGDClassifier(
        ...         loss='log',
        ...         eta0=0.01,
        ...         learning_rate='constant'
        ...     ),
        ...     classes=[False, True]
        ... )

        >>> metric = metrics.LogLoss()

        >>> model_selection.progressive_val_score(X_y, model, metric)
        LogLoss: 0.199554

    """

    def __init__(self, estimator: sklearn_base.ClassifierMixin, classes: list, batch_size=1):
        super().__init__(
            estimator=estimator,
            batch_size=batch_size,
            x_dtype=np.float
        )
        self.classes = classes

    def fit_one(self, x, y):

        if self._x_batch is None:
            n_features = len(x)
            self._x_batch = np.empty(shape=(self.batch_size, n_features), dtype=self.x_dtype)
            self._y_batch = [None] * self.batch_size

        self._x_batch[self._batch_i, :] = list(x.values())
        self._y_batch[self._batch_i] = y

        self._batch_i += 1

        if self._batch_i == self.batch_size:
            self.estimator.partial_fit(X=self._x_batch, y=self._y_batch, classes=self.classes)
            self._batch_i = 0

        return self

    def predict_proba_one(self, x):
        try:
            y_pred = self.estimator.predict_proba([list(x.values())])[0]
            return {self.classes[i]: p for i, p in enumerate(y_pred)}
        except exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_one(self, x):
        try:
            y_pred = self.estimator.predict(X=[list(x.values())])[0]
            return y_pred
        except exceptions.NotFittedError:
            return self.classes[0]
