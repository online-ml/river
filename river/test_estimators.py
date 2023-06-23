"""General tests that all estimators need to pass."""
from __future__ import annotations

import importlib
import inspect

import pytest

from river import (
    anomaly,
    base,
    checks,
    compat,
    compose,
    facto,
    feature_extraction,
    feature_selection,
    imblearn,
    linear_model,
    model_selection,
    multiclass,
    neighbors,
    neural_net,
    preprocessing,
    time_series,
)

try:
    from river.compat.pytorch import PyTorch2RiverBase

    PYTORCH_INSTALLED = True
except ImportError:
    PYTORCH_INSTALLED = False
from sklearn import linear_model as sk_linear_model

from river.compat.river_to_sklearn import River2SKLBase
from river.compat.sklearn_to_river import SKL2RiverBase


def iter_estimators():
    for submodule in importlib.import_module("river.api").__all__:

        def is_estimator(obj):
            return inspect.isclass(obj) and issubclass(obj, base.Estimator)

        for _, obj in inspect.getmembers(
            importlib.import_module(f"river.{submodule}"), is_estimator
        ):
            yield obj


def iter_estimators_which_can_be_tested():
    ignored = (
        River2SKLBase,
        SKL2RiverBase,
        compose.FuncTransformer,
        compose.Grouper,
        compose.Pipeline,
        compose.Prefixer,
        compose.Renamer,
        compose.Suffixer,
        compose.TargetTransformRegressor,
        facto.FFMClassifier,
        facto.FFMRegressor,
        facto.FMClassifier,
        facto.FMRegressor,
        facto.FwFMClassifier,
        facto.FwFMRegressor,
        facto.HOFMClassifier,
        facto.HOFMRegressor,
        feature_extraction.Agg,
        feature_extraction.TargetAgg,
        feature_selection.PoissonInclusion,
        imblearn.RandomOverSampler,
        imblearn.RandomUnderSampler,
        imblearn.RandomSampler,
        model_selection.SuccessiveHalvingClassifier,
        neighbors.LazySearch,
        neural_net.MLPRegressor,
        preprocessing.PreviousImputer,
        preprocessing.OneHotEncoder,
        preprocessing.StatImputer,
        time_series.base.Forecaster,
    )

    if PYTORCH_INSTALLED:
        ignored = (*ignored, PyTorch2RiverBase)

    def can_be_tested(estimator):
        return not inspect.isabstract(estimator) and not issubclass(estimator, ignored)

    for estimator in filter(can_be_tested, iter_estimators()):
        for params in estimator._unit_test_params():
            yield estimator(**params)


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(
            estimator,
            check,
            id=f"{estimator}:{check.__name__}",
        )
        for estimator in list(iter_estimators_which_can_be_tested())
        + [
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            preprocessing.StandardScaler() | linear_model.PAClassifier(),
            (
                preprocessing.StandardScaler()
                | preprocessing.TargetStandardScaler(
                    regressor=linear_model.LinearRegression(),
                )
            ),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
            ),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.PAClassifier())
            ),
            preprocessing.MinMaxScaler() + preprocessing.StandardScaler(),
            (
                feature_extraction.PolynomialExtender()
                | preprocessing.StandardScaler()
                | linear_model.LinearRegression()
            ),
            preprocessing.MinMaxScaler() | anomaly.HalfSpaceTrees(),
            (
                preprocessing.StandardScaler()
                | compat.convert_sklearn_to_river(sk_linear_model.SGDRegressor(tol=1e-10))
            ),
        ]
        for check in checks.yield_checks(estimator)
        if check.__name__ not in estimator._unit_test_skips()
    ],
)
def test_check_estimator(estimator, check):
    check(estimator.clone())
