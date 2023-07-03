from __future__ import annotations

import copy
import functools

import pandas as pd
from sklearn import base as sklearn_base
from sklearn import exceptions as sklearn_exceptions

from river import base

__all__ = ["convert_sklearn_to_river", "SKL2RiverClassifier", "SKL2RiverRegressor"]


def convert_sklearn_to_river(estimator: sklearn_base.BaseEstimator, classes: list | None = None):
    """Wraps a scikit-learn estimator to make it compatible with river.

    Parameters
    ----------
    estimator
    classes
        Class names necessary for classifiers.

    """

    if not hasattr(estimator, "partial_fit"):
        raise ValueError(f"{estimator} does not have a partial_fit method")

    if isinstance(estimator, sklearn_base.ClassifierMixin) and classes is None:
        raise ValueError("classes must be provided to convert a classifier")

    wrappers = [
        (sklearn_base.RegressorMixin, SKL2RiverRegressor),
        (
            sklearn_base.ClassifierMixin,
            functools.partial(SKL2RiverClassifier, classes=classes),
        ),
    ]

    for base_type, wrapper in wrappers:
        if isinstance(estimator, base_type):
            return wrapper(copy.deepcopy(estimator))  # type: ignore

    raise ValueError("Couldn't find an appropriate wrapper")


class SKL2RiverBase:
    def __init__(self, estimator: sklearn_base.BaseEstimator):
        self.estimator = estimator
        self._feature_names: list | None = None

    def _align_dict(self, x: dict) -> list:
        if self._feature_names is None:
            self._feature_names = list(x.keys())
        return [x[k] for k in self._feature_names]

    def _align_df(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._feature_names is None:
            self._feature_names = list(X.columns)
        return X[self._feature_names]

    def _unit_test_skips(self):  # noqa
        return {
            "check_emerging_features",
            "check_disappearing_features",
        }


class SKL2RiverRegressor(SKL2RiverBase, base.Regressor):
    """Compatibility layer from scikit-learn to River for regression.

    Parameters
    ----------
    estimator
        A scikit-learn transformer which has a `partial_fit` method.

    Examples
    --------

    >>> from river import compat
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import linear_model
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_diabetes(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> scaler = preprocessing.StandardScaler()
    >>> sgd_reg = compat.convert_sklearn_to_river(linear_model.SGDRegressor())
    >>> model = scaler | sgd_reg

    >>> metric = metrics.MAE()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 84.501421

    """

    def learn_one(self, x, y):
        self.estimator.partial_fit(X=[self._align_dict(x)], y=[y])
        return self

    def learn_many(self, X, y):
        self.estimator.partial_fit(X=self._align_df(X), y=y)
        return self

    def predict_one(self, x):
        try:
            return self.estimator.predict(X=[self._align_dict(x)])[0]
        except sklearn_exceptions.NotFittedError:
            return 0

    def predict_many(self, X):
        try:
            return pd.Series(self.estimator.predict(self._align_df(X)))
        except sklearn_exceptions.NotFittedError:
            return pd.Series([0] * len(X), index=X.index)


class SKL2RiverClassifier(SKL2RiverBase, base.Classifier):
    """Compatibility layer from scikit-learn to River for classification.

    Parameters
    ----------
    estimator
        A scikit-learn regressor which has a `partial_fit` method.
    classes

    Examples
    --------

    >>> from river import compat
    >>> from river import evaluate
    >>> from river import metrics
    >>> from river import preprocessing
    >>> from river import stream
    >>> from sklearn import linear_model
    >>> from sklearn import datasets

    >>> dataset = stream.iter_sklearn_dataset(
    ...     dataset=datasets.load_breast_cancer(),
    ...     shuffle=True,
    ...     seed=42
    ... )

    >>> model = preprocessing.StandardScaler()
    >>> model |= compat.convert_sklearn_to_river(
    ...     estimator=linear_model.SGDClassifier(
    ...         loss='log_loss',
    ...         eta0=0.01,
    ...         learning_rate='constant'
    ...     ),
    ...     classes=[False, True]
    ... )

    >>> metric = metrics.LogLoss()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    LogLoss: 0.198029

    """

    def __init__(self, estimator: sklearn_base.ClassifierMixin, classes: list):
        super().__init__(estimator)
        self.classes = classes

    @property
    def _multiclass(self):
        return True

    def learn_one(self, x, y):
        self.estimator.partial_fit(X=[self._align_dict(x)], y=[y], classes=self.classes)
        return self

    def learn_many(self, X, y):
        self.estimator.partial_fit(X=self._align_df(X), y=y, classes=self.classes)
        return self

    def predict_proba_one(self, x):
        try:
            y_pred = self.estimator.predict_proba([self._align_dict(x)])[0]
            return {self.classes[i]: p for i, p in enumerate(y_pred)}
        except sklearn_exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_proba_many(self, X):
        try:
            return pd.Series(self.estimator.predict_proba(self._align_df(X)), columns=self.classes)
        except sklearn_exceptions.NotFittedError:
            return pd.DataFrame(
                [[1 / len(self.classes)] * len(self.classes)] * len(X),
                columns=self.classes,
                index=X.index,
            )

    def predict_one(self, x):
        try:
            y_pred = self.estimator.predict(X=[self._align_dict(x)])[0]
            return y_pred
        except sklearn_exceptions.NotFittedError:
            return self.classes[0]

    def predict_many(self, X):
        try:
            return pd.Series(self.estimator.predict(self._align_df(X)))
        except sklearn_exceptions.NotFittedError:
            return pd.Series([self.classes[0]] * len(X), index=X.index)
