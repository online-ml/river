"""General tests that all estimators need to pass."""
import copy
import importlib
import inspect

import pytest

from river import (
    base,
    cluster,
    compat,
    compose,
    ensemble,
    expert,
    facto,
    feature_extraction,
    feature_selection,
    imblearn,
    linear_model,
    meta,
    multiclass,
    naive_bayes,
    neural_net,
    preprocessing,
    reco,
    stats,
    time_series,
    utils,
)
from river.compat.river_to_sklearn import River2SKLBase
from river.compat.sklearn_to_river import SKL2RiverBase


def get_all_estimators():

    ignored = (
        River2SKLBase,
        SKL2RiverBase,
        compose.FuncTransformer,
        compose.Pipeline,
        compose.Grouper,
        ensemble.AdaptiveRandomForestClassifier,
        ensemble.AdaptiveRandomForestRegressor,
        ensemble.SRPClassifier,
        expert.StackingClassifier,
        expert.SuccessiveHalvingClassifier,
        expert.SuccessiveHalvingRegressor,
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
        meta.PredClipper,
        meta.TransformedTargetRegressor,
        neural_net.MLPRegressor,
        preprocessing.PreviousImputer,
        preprocessing.OneHotEncoder,
        preprocessing.StatImputer,
        reco.Baseline,
        reco.BiasedMF,
        reco.FunkMF,
        reco.RandomNormal,
        imblearn.RandomOverSampler,
        imblearn.RandomUnderSampler,
        imblearn.RandomSampler,
        time_series.Detrender,
        time_series.GroupDetrender,
        time_series.SNARIMAX,
    )

    try:
        ignored = (*ignored, compat.PyTorch2RiverRegressor)
    except AttributeError:
        pass

    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, base.Estimator)

    for submodule in importlib.import_module("river").__all__:

        if submodule == "base":
            continue

        if submodule == "synth":
            submodule = "datasets.synth"

        submodule = f"river.{submodule}"

        for _, obj in inspect.getmembers(
            importlib.import_module(submodule), is_estimator
        ):
            if issubclass(obj, ignored):
                continue
            params = obj._unit_test_params()
            yield obj(**params)


@pytest.mark.parametrize(
    "estimator, check",
    [
        pytest.param(estimator, check, id=f"{estimator}:{check.__name__}")
        for estimator in list(get_all_estimators())
        + [
            feature_extraction.TFIDF(),
            linear_model.LogisticRegression(),
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            preprocessing.StandardScaler() | linear_model.PAClassifier(),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
            ),
            (
                preprocessing.StandardScaler()
                | multiclass.OneVsRestClassifier(linear_model.PAClassifier())
            ),
            naive_bayes.GaussianNB(),
            preprocessing.StandardScaler(),
            cluster.KMeans(n_clusters=5, seed=42),
            preprocessing.MinMaxScaler(),
            preprocessing.MinMaxScaler() + preprocessing.StandardScaler(),
            feature_extraction.PolynomialExtender(),
            (
                feature_extraction.PolynomialExtender()
                | preprocessing.StandardScaler()
                | linear_model.LinearRegression()
            ),
            feature_selection.VarianceThreshold(),
            feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
        ]
        for check in utils.estimator_checks.yield_checks(estimator)
        if check.__name__ not in estimator._unit_test_skips()
    ],
)
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))
