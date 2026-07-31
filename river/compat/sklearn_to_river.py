from __future__ import annotations

import copy
import typing

import narwhals.stable.v2 as nw
import numpy as np
from sklearn import base as sklearn_base
from sklearn import exceptions as sklearn_exceptions
from sklearn import linear_model as sklearn_linear_model

from river import base, utils

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals.stable.v2.typing import IntoDataFrame, IntoDataFrameT, IntoSeries
    from numpy.typing import NDArray

    T = typing.TypeVar("T")

    class _IncrementalEstimator(typing.Protocol):
        """The duck-typed slice of a scikit-learn estimator these wrappers rely on.

        `sklearn.base.BaseEstimator` declares none of these: `partial_fit`/`predict` come from the
        mixins and are only present on the concrete estimators. Naming the contract here keeps the
        call sites typed, and mirrors the `hasattr(estimator, "partial_fit")` guard in
        `convert_sklearn_to_river`.
        """

        def partial_fit(self, X: typing.Any, y: typing.Any, **kwargs: typing.Any) -> typing.Any: ...

        def predict(self, X: typing.Any) -> NDArray[typing.Any]: ...

    class _ProbabilisticEstimator(_IncrementalEstimator, typing.Protocol):
        def predict_proba(self, X: typing.Any) -> NDArray[np.float64]: ...


__all__ = ["convert_sklearn_to_river", "SKL2RiverClassifier", "SKL2RiverRegressor"]


@typing.overload
def convert_sklearn_to_river(
    estimator: sklearn_base.BaseEstimator, classes: None = None
) -> SKL2RiverRegressor: ...


@typing.overload
def convert_sklearn_to_river(
    estimator: sklearn_base.BaseEstimator, classes: list[base.typing.ClfTarget]
) -> SKL2RiverClassifier: ...


def convert_sklearn_to_river(
    estimator: sklearn_base.BaseEstimator, classes: list[base.typing.ClfTarget] | None = None
) -> SKL2RiverRegressor | SKL2RiverClassifier:
    """Wraps a scikit-learn estimator to make it compatible with river.

    Parameters
    ----------
    estimator
    classes
        Class names. Required for classifiers, and not accepted for regressors.

    """

    if not hasattr(estimator, "partial_fit"):
        raise ValueError(f"{estimator} does not have a partial_fit method")

    if isinstance(estimator, sklearn_base.RegressorMixin):
        if classes is not None:
            raise ValueError("classes is only used for classifiers, a regressor cannot take it")
        return SKL2RiverRegressor(copy.deepcopy(estimator))

    if isinstance(estimator, sklearn_base.ClassifierMixin):
        if classes is None:
            raise ValueError("classes must be provided to convert a classifier")
        return SKL2RiverClassifier(copy.deepcopy(estimator), classes=classes)

    raise ValueError("Couldn't find an appropriate wrapper")


class SKL2RiverBase:
    def __init__(self, estimator: sklearn_base.BaseEstimator) -> None:
        # The public contract is "any scikit-learn estimator", but only the incremental slice of
        # its interface is ever used; see `_IncrementalEstimator`.
        self.estimator = typing.cast("_IncrementalEstimator", estimator)
        self._feature_names: list[base.typing.FeatureName] | None = None

    def _align_dict(self, x: dict[base.typing.FeatureName, T]) -> list[T]:
        if self._feature_names is None:
            self._feature_names = list(x.keys())
        return [x[k] for k in self._feature_names]

    def _align_frame(self, X: IntoDataFrameT) -> nw.DataFrame[IntoDataFrameT]:
        X_nw = utils.dataframe.into_frame(X)
        columns = X_nw.columns
        if self._feature_names is None:
            self._feature_names = list(columns)
        if columns == self._feature_names:
            return X_nw
        if missing := [name for name in self._feature_names if name not in columns]:
            raise ValueError(
                f"The following features are missing from the mini-batch: {missing}. "
                "Every batch has to carry the features seen in the first one."
            )
        # `nw.col` rather than bare names: pandas allows non-string column labels
        # (a frame built from an array is keyed on integers), which `select` rejects
        # unless wrapped with nw.col(...)
        return X_nw.select(nw.col(self._feature_names))  # type: ignore[arg-type]

    def _unit_test_skips(self) -> set[str]:
        return {
            "check_emerging_features",
            "check_disappearing_features",
            "check_radically_disappearing_features",
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

    def learn_one(
        self, x: dict[base.typing.FeatureName, typing.Any], y: base.typing.RegTarget
    ) -> None:
        self.estimator.partial_fit(X=[self._align_dict(x)], y=[y])

    def learn_many(self, X: IntoDataFrame, y: IntoSeries) -> None:
        self.estimator.partial_fit(X=self._align_frame(X).to_native(), y=y)

    def predict_one(self, x: dict[base.typing.FeatureName, typing.Any]) -> base.typing.RegTarget:
        try:
            prediction = self.estimator.predict(X=[self._align_dict(x)])[0]
        except sklearn_exceptions.NotFittedError:
            return 0
        # Indexing a numpy array yields `typing.Any`; a regressor predicts a real number.
        return typing.cast("base.typing.RegTarget", prediction)

    def predict_many(self, X: IntoDataFrame) -> IntoSeries:
        X_nw = self._align_frame(X)
        values: NDArray[typing.Any]
        try:
            values = self.estimator.predict(X_nw.to_native())
        except sklearn_exceptions.NotFittedError:
            # Mirror the `0` that `predict_one` falls back on, dtype included.
            values = np.zeros(len(X_nw), dtype=np.int64)
        return utils.dataframe.to_native_series(values, name=None, like=X_nw)

    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, typing.Any]]:
        yield {"estimator": sklearn_linear_model.SGDRegressor()}


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

    #: A classifier additionally has to expose `predict_proba`; this narrows the declaration
    #: inherited from `SKL2RiverBase`.
    estimator: _ProbabilisticEstimator

    def __init__(
        self, estimator: sklearn_base.BaseEstimator, classes: list[base.typing.ClfTarget]
    ) -> None:
        super().__init__(estimator)
        self.classes = classes

    @property
    def _multiclass(self) -> bool:
        return len(self.classes) > 2

    def learn_one(
        self, x: dict[base.typing.FeatureName, typing.Any], y: base.typing.ClfTarget
    ) -> None:
        self.estimator.partial_fit(X=[self._align_dict(x)], y=[y], classes=self.classes)

    def learn_many(self, X: IntoDataFrame, y: IntoSeries) -> None:
        self.estimator.partial_fit(X=self._align_frame(X).to_native(), y=y, classes=self.classes)

    def predict_proba_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> dict[base.typing.ClfTarget, float]:
        try:
            y_pred = self.estimator.predict_proba([self._align_dict(x)])[0]
            return {self.classes[i]: float(p) for i, p in enumerate(y_pred)}
        except sklearn_exceptions.NotFittedError:
            return {c: 1 / len(self.classes) for c in self.classes}

    def predict_proba_many(self, X: IntoDataFrame) -> IntoDataFrame:
        X_nw = self._align_frame(X)
        probas: NDArray[np.float64]
        try:
            probas = self.estimator.predict_proba(X_nw.to_native())
        except sklearn_exceptions.NotFittedError:
            probas = np.full((len(X_nw), len(self.classes)), 1 / len(self.classes))
        return utils.dataframe.to_native_frame(probas, like=X_nw, columns=self.classes)

    def predict_one(
        self, x: dict[base.typing.FeatureName, typing.Any], **kwargs: typing.Any
    ) -> base.typing.ClfTarget:
        try:
            prediction = self.estimator.predict(X=[self._align_dict(x)])[0]
        except sklearn_exceptions.NotFittedError:
            return self.classes[0]
        # Indexing a numpy array yields `typing.Any`; the values are the labels the model was given.
        return typing.cast("base.typing.ClfTarget", prediction)

    def predict_many(self, X: IntoDataFrame) -> IntoSeries:
        X_nw = self._align_frame(X)
        values: NDArray[typing.Any] | list[base.typing.ClfTarget]
        try:
            values = self.estimator.predict(X_nw.to_native())
        except sklearn_exceptions.NotFittedError:
            # Mirror the first class that `predict_one` falls back on.
            values = [self.classes[0]] * len(X_nw)
        return utils.dataframe.to_native_series(values, name=None, like=X_nw)

    @classmethod
    def _unit_test_params(cls) -> Iterator[dict[str, typing.Any]]:
        yield {
            "estimator": sklearn_linear_model.SGDClassifier(loss="log_loss"),
            "classes": [False, True],
        }
