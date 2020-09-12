import copy
import functools
import typing

import numpy as np
import pandas as pd
from sklearn import base as sklearn_base
from sklearn import exceptions

from creme import base


__all__ = [
    'convert_sklearn_to_creme',
    'SKL2CremeClassifier',
    'SKL2CremeRegressor'
]


def convert_sklearn_to_creme(estimator: sklearn_base.BaseEstimator, classes: list = None):
    """Wraps a scikit-learn estimator to make it compatible with creme.

    Parameters
    ----------
    estimator
    classes
        Class names necessary for classifiers.

    """

    if not hasattr(estimator, 'partial_fit'):
        raise ValueError(f'{estimator} does not have a partial_fit method')

    if isinstance(estimator, sklearn_base.ClassifierMixin) and classes is None:
        raise ValueError('classes must be provided to convert a classifier')

    wrappers = [
        (sklearn_base.RegressorMixin, SKL2CremeRegressor),
        (sklearn_base.ClassifierMixin, functools.partial(
            SKL2CremeClassifier,
            classes=classes
        ))
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            return wrapper(copy.deepcopy(estimator))

    raise ValueError("Couldn't find an appropriate wrapper")


class SKL2CremeBase:

    def __init__(self, estimator: sklearn_base.BaseEstimator):
        self.estimator = estimator


class SKL2CremeRegressor(SKL2CremeBase, base.Regressor):
    """Converts a `sklearn` regressor to a `creme` regressor.

    Parameters
    ----------
    estimator
        A scikit-learn transformer which has a `partial_fit` method.

    Examples
    --------

    >>> from creme import compat
    >>> from creme import evaluate
    >>> from creme import metrics
    >>> from creme import preprocessing
    >>> from creme import stream
    >>> from sklearn import linear_model
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_boston(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> scaler = preprocessing.StandardScaler()
    >>> sgd_reg = compat.convert_sklearn_to_creme(linear_model.SGDRegressor())
    >>> model = scaler | sgd_reg

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 11.004415

    """

    def learn_one(self, x, y):
        self.estimator.partial_fit(X=[list(x.values())], y=[y])
        return self

    def learn_many(self, X, y):
        self.estimator.partial_fit(X=X.values, y=y.values)
        return self

    def predict_one(self, x):
        try:
            return self.estimator.predict(X=[list(x.values())])[0]
        except exceptions.NotFittedError:
            return 0

    def predict_many(self, X):
        return pd.Series(self.estimator.predict(X))


class SKL2CremeClassifier(SKL2CremeBase, base.Classifier):
    """Converts a `sklearn` classifier to `creme` classifier.

    Parameters
    ----------
    estimator
        A scikit-learn regressor which has a `partial_fit` method.
    classes

    Examples
    --------

    >>> from creme import compat
    >>> from creme import evaluate
    >>> from creme import metrics
    >>> from creme import preprocessing
    >>> from creme import stream
    >>> from sklearn import linear_model
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
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

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.199554

    """

    def __init__(self, estimator: sklearn_base.ClassifierMixin, classes: list):
        super().__init__(estimator)
        self.classes = classes

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        self.estimator.partial_fit(X=[list(x.values())], y=[y], classes=self.classes)
        return self

    def learn_many(self, X, y):
        self.estimator.partial_fit(X=X.values, y=y.values, classes=self.classes)
        return self

    def predict_proba_one(self, x):
        try:
            y_pred = self.estimator.predict_proba([list(x.values())])[0]
            return {self.classes[i]: p for i, p in enumerate(y_pred)}
        except exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_proba_many(self, X):
        return pd.Series(self.estimator.predict_proba(X), columns=self.classes)

    def predict_one(self, x):
        try:
            y_pred = self.estimator.predict(X=[list(x.values())])[0]
            return y_pred
        except exceptions.NotFittedError:
            return self.classes[0]

    def predict_many(self, X):
        return pd.Series(self.estimator.predict(X))
