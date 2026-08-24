from __future__ import annotations

import typing

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets as sk_datasets
from sklearn import linear_model as sk_linear_model
from sklearn.utils import estimator_checks

from river import base, cluster, compat, linear_model, preprocessing
from tests.frames import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries
    from numpy.typing import NDArray

    from river import compose
    from tests.frames import FrameBackend

    WrappedRegressor = compat.SKL2RiverRegressor | compose.Pipeline
    WrappedClassifier = compat.SKL2RiverClassifier | compose.Pipeline

    ClassifierReference = tuple[list[typing.Any], NDArray[np.float64]]
    """The pandas oracle a classifier's mini-batch output is compared against;
    consisting of (labels, probability matrix). """


def _regressor() -> compat.SKL2RiverRegressor:
    return compat.convert_sklearn_to_river(sk_linear_model.SGDRegressor(random_state=42))


def _classifier(n_classes: int = 2) -> compat.SKL2RiverClassifier:
    return compat.convert_sklearn_to_river(
        sk_linear_model.SGDClassifier(loss="log_loss", random_state=42),
        classes=list(range(n_classes)),
    )


def _labels(series: IntoSeries) -> list[typing.Any]:
    return nw.from_native(series, series_only=True).to_list()


def _values(frame: IntoDataFrame) -> NDArray[np.float64]:
    """Read a native frame into a float matrix, positionally."""
    return np.asarray(nw.from_native(frame, eager_only=True).to_numpy(), dtype=np.float64)


def _implementation(native: IntoDataFrame | IntoSeries) -> nw.Implementation:
    """The dataframe library a native frame or series belongs to."""
    return nw.from_native(native, allow_series=True).implementation


@pytest.mark.parametrize(
    "estimator",
    [
        linear_model.LinearRegression(),
        linear_model.LogisticRegression(),
        preprocessing.StandardScaler(),
        cluster.KMeans(n_clusters=3, seed=42),
    ],
    ids=str,
)
@pytest.mark.filterwarnings("ignore::sklearn.utils.estimator_checks.SkipTestWarning")
def test_river_to_sklearn_check_estimator(estimator: base.Estimator) -> None:
    skl_estimator = compat.convert_river_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)


@pytest.mark.filterwarnings("ignore::sklearn.utils.estimator_checks.SkipTestWarning")
def test_sklearn_check_twoway() -> None:
    estimator = sk_linear_model.SGDRegressor()
    river_estimator = compat.convert_sklearn_to_river(estimator)
    skl_estimator = compat.convert_river_to_sklearn(river_estimator)
    estimator_checks.check_estimator(skl_estimator)


@pytest.mark.parametrize(
    "estimator",
    [_regressor(), preprocessing.StandardScaler() | _regressor()],
    ids=str,
)
def test_not_fitted_still_works_regression(estimator: WrappedRegressor) -> None:
    _Xs, _ = sk_datasets.make_regression(n_samples=500, n_features=4)
    X = pd.DataFrame(_Xs)

    y_pred = estimator.predict_many(X)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(X)
    assert (y_pred == 0).all()


@pytest.mark.parametrize(
    "estimator,n_classes",
    [
        pytest.param(estimator, n_classes, id=f"{estimator}-{n_classes} classes")
        for n_classes in [2, 3]
        for estimator in [
            _classifier(n_classes),
            preprocessing.StandardScaler() | _classifier(n_classes),
        ]
    ],
)
def test_not_fitted_still_works_classification(
    estimator: WrappedClassifier, n_classes: int
) -> None:
    _Xs, _ys = sk_datasets.make_classification(
        n_samples=500, n_features=10, n_informative=6, n_classes=n_classes
    )
    X = pd.DataFrame(_Xs)
    y = pd.Series(_ys)

    y_pred = estimator.predict_many(X)
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(X)
    assert (y_pred == 0).all()

    y_pred_proba = estimator.predict_proba_many(X)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert y_pred_proba.shape == (len(X), n_classes)

    # Also exercise the fitted path of predict_proba_many.
    estimator.learn_many(X, y)
    y_pred_proba = estimator.predict_proba_many(X)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert y_pred_proba.shape == (len(X), n_classes)
    assert list(y_pred_proba.index) == list(X.index)


# The `*_many` methods of the scikit-learn wrappers go through narwhals, so any eager backend
# (pandas, polars, pyarrow, ...) can be fed in and comes back out.

Xs: dict[str, list[float]] = {
    "f0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "f1": [1.0, 0.5, 0.0, -0.5, -1.0, -1.5],
}
Ys = [0, 0, 0, 1, 1, 1]


@pytest.fixture(scope="module")
def regressor_reference() -> list[typing.Any]:
    """Predictions of a pandas-fed regressor: the oracle every other backend has to match."""
    pandas = FRAME_BACKENDS["pandas"]()
    model = _regressor()
    model.learn_many(pandas.frame(Xs), pandas.series(Ys))
    return _labels(model.predict_many(pandas.frame(Xs)))


@pytest.fixture(scope="module")
def classifier_reference() -> ClassifierReference:
    """Labels and probabilities of a pandas-fed classifier: the oracle for the other backends."""
    pandas = FRAME_BACKENDS["pandas"]()
    model = _classifier()
    model.learn_many(pandas.frame(Xs), pandas.series(Ys))
    X = pandas.frame(Xs)
    return _labels(model.predict_many(X)), _values(model.predict_proba_many(X))


def test_sklearn_regressor_mini_batch_is_backend_agnostic(
    frame_backend: FrameBackend, regressor_reference: list[typing.Any]
) -> None:
    X = frame_backend.frame(Xs)
    model = _regressor()
    model.learn_many(X, frame_backend.series(Ys))

    preds = model.predict_many(X)

    assert _implementation(preds) == _implementation(X)
    np.testing.assert_allclose(_labels(preds), regressor_reference, rtol=1e-12)


def test_sklearn_classifier_mini_batch_is_backend_agnostic(
    frame_backend: FrameBackend, classifier_reference: ClassifierReference
) -> None:
    expected_labels, expected_proba = classifier_reference

    X = frame_backend.frame(Xs)
    model = _classifier()
    model.learn_many(X, frame_backend.series(Ys))

    preds = model.predict_many(X)
    proba = model.predict_proba_many(X)

    assert _implementation(preds) == _implementation(X)
    assert _implementation(proba) == _implementation(X)
    assert _labels(preds) == expected_labels
    np.testing.assert_allclose(_values(proba), expected_proba, rtol=1e-12)


def test_sklearn_wrapper_realigns_reordered_columns(frame_backend: FrameBackend) -> None:
    """scikit-learn identifies features by position, so later batches are reordered to match."""
    X = frame_backend.frame(Xs)
    model = _regressor()
    model.learn_many(X, frame_backend.series(Ys))

    in_order = model.predict_many(X)
    reordered = model.predict_many(
        frame_backend.frame({key: Xs[key] for key in reversed(list(Xs))})
    )

    np.testing.assert_allclose(_labels(reordered), _labels(in_order))


def test_sklearn_wrapper_realigns_reordered_integer_columns() -> None:
    """Pandas is the only backend accepting non-string labels, as a frame built from an array has."""
    X = pd.DataFrame(np.column_stack(list(Xs.values())))
    assert list(X.columns) == [0, 1]

    model = _regressor()
    model.learn_many(X, pd.Series(Ys))

    in_order = model.predict_many(X)
    reordered = model.predict_many(X[list(reversed(X.columns))])

    np.testing.assert_allclose(_labels(reordered), _labels(in_order))


def test_sklearn_wrapper_rejects_a_missing_feature(frame_backend: FrameBackend) -> None:
    model = _regressor()
    model.learn_many(frame_backend.frame(Xs), frame_backend.series(Ys))

    with pytest.raises(ValueError, match="missing from the mini-batch"):
        model.predict_many(frame_backend.frame({"f0": Xs["f0"]}))
