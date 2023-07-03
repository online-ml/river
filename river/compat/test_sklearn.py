from __future__ import annotations

import pandas as pd
import pytest
from sklearn import datasets as sk_datasets
from sklearn import linear_model as sk_linear_model
from sklearn.utils import estimator_checks

from river import base, cluster, compat, linear_model, preprocessing


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(estimator, id=str(estimator))
        for estimator in [
            linear_model.LinearRegression(),
            linear_model.LogisticRegression(),
            preprocessing.StandardScaler(),
            cluster.KMeans(n_clusters=3, seed=42),
        ]
    ],
)
@pytest.mark.filterwarnings("ignore::sklearn.utils.estimator_checks.SkipTestWarning")
def test_river_to_sklearn_check_estimator(estimator: base.Estimator):
    skl_estimator = compat.convert_river_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)


@pytest.mark.filterwarnings("ignore::sklearn.utils.estimator_checks.SkipTestWarning")
def test_sklearn_check_twoway():
    estimator = sk_linear_model.SGDRegressor()
    river_estimator = compat.convert_sklearn_to_river(estimator)
    skl_estimator = compat.convert_river_to_sklearn(river_estimator)
    estimator_checks.check_estimator(skl_estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(estimator, id=str(estimator))
        for estimator in [
            compat.convert_sklearn_to_river(sk_linear_model.SGDRegressor()),
            (
                preprocessing.StandardScaler()
                | compat.convert_sklearn_to_river(sk_linear_model.SGDRegressor())
            ),
        ]
    ],
)
def test_not_fitted_still_works_regression(estimator):
    X, _ = sk_datasets.make_regression(n_samples=500, n_features=4)
    X = pd.DataFrame(X)
    y_pred = estimator.predict_many(X)
    assert len(y_pred) == len(X)
    assert y_pred.eq(0).all()


@pytest.mark.parametrize(
    "estimator,n_classes",
    [
        pytest.param(estimator, n_classes, id=f"{estimator}-{n_classes} classes")
        for n_classes in [2, 3]
        for estimator in [
            compat.convert_sklearn_to_river(
                sk_linear_model.SGDClassifier(loss="log_loss"), classes=list(range(n_classes))
            ),
            (
                preprocessing.StandardScaler()
                | compat.convert_sklearn_to_river(
                    sk_linear_model.SGDClassifier(loss="log_loss"), classes=list(range(n_classes))
                )
            ),
        ]
    ],
)
def test_not_fitted_still_works_classification(estimator, n_classes):
    X, _ = sk_datasets.make_classification(
        n_samples=500, n_features=10, n_informative=6, n_classes=n_classes
    )
    X = pd.DataFrame(X)
    y_pred = estimator.predict_many(X)
    assert len(y_pred) == len(X)
    assert y_pred.eq(0).all()

    y_pred_proba = estimator.predict_proba_many(X)
    assert y_pred_proba.shape == (len(X), n_classes)
