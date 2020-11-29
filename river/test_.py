"""General tests that all estimators need to pass."""
import copy
import importlib
import inspect

import pytest

from river import base
from river import compat
from river import cluster
from river import compose
from river import ensemble
from river import expert
from river import facto
from river import feature_extraction
from river import feature_selection
from river import imblearn
from river import linear_model
from river import meta
from river import multiclass
from river import naive_bayes
from river import optim
from river import preprocessing
from river import reco
from river import stats
from river import time_series
from river import utils
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
        linear_model.SoftmaxRegression,
        meta.PredClipper,
        meta.TransformedTargetRegressor,
        preprocessing.PreviousImputer,
        preprocessing.OneHotEncoder,
        preprocessing.StatImputer,
        reco.Baseline,
        reco.BiasedMF,
        reco.FunkMF,
        reco.RandomNormal,
        imblearn.HardSamplingClassifier,
        imblearn.HardSamplingRegressor,
        imblearn.RandomOverSampler,
        imblearn.RandomUnderSampler,
        imblearn.RandomSampler,
        time_series.Detrender,
        time_series.GroupDetrender,
        time_series.SNARIMAX
    )

    try:
        ignored = (*ignored, compat.PyTorch2RiverRegressor)
    except AttributeError:
        pass

    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, base.Estimator)

    for submodule in importlib.import_module('river').__all__:

        if submodule == 'base':
            continue

        if submodule == 'synth':
            submodule = 'datasets.synth'

        for _, obj in inspect.getmembers(importlib.import_module(f'river.{submodule}'), is_estimator):
            if issubclass(obj, ignored):
                continue
            try:
                params = obj._default_params()
            except AttributeError:
                params = {}
            yield obj(**params)


@pytest.mark.parametrize('estimator, check', [
    pytest.param(
        model,
        check,
        id=f'{model}:{check.__name__}'
    )
    for model in list(get_all_estimators()) + [
        feature_extraction.TFIDF(),
        linear_model.LogisticRegression(),
        preprocessing.StandardScaler() | linear_model.LinearRegression(),
        preprocessing.StandardScaler() | linear_model.PAClassifier(),
        preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(linear_model.LogisticRegression()),
        preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(linear_model.PAClassifier()),
        naive_bayes.GaussianNB(),
        preprocessing.StandardScaler(),
        cluster.KMeans(n_clusters=5, seed=42),
        preprocessing.MinMaxScaler(),
        preprocessing.MinMaxScaler() + preprocessing.StandardScaler(),
        feature_extraction.PolynomialExtender(),
        (
            feature_extraction.PolynomialExtender() |
            preprocessing.StandardScaler() |
            linear_model.LinearRegression()
        ),
        feature_selection.VarianceThreshold(),
        feature_selection.SelectKBest(similarity=stats.PearsonCorr()),
        (
            preprocessing.StandardScaler() |
            linear_model.LogisticRegression(
                optimizer=optim.Adam(),
                initializer=optim.initializers.Normal(seed=42),
                l2=.1
            )
        )
    ]
    for check in utils.estimator_checks.yield_checks(model)
])
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))
