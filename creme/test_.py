"""General tests that all estimators need to pass."""
import copy
import importlib
import inspect

import pytest

from creme import base
from creme import compat
from creme import cluster
from creme import compose
from creme import expert
from creme import facto
from creme import feature_extraction
from creme import feature_selection
from creme import imblearn
from creme import linear_model
from creme import meta
from creme import multiclass
from creme import naive_bayes
from creme import preprocessing
from creme import reco
from creme import stats
from creme import time_series
from creme import utils
from creme.compat.creme_to_sklearn import Creme2SKLBase
from creme.compat.sklearn_to_creme import SKL2CremeBase


def get_all_estimators():

    ignored = (
        Creme2SKLBase,
        SKL2CremeBase,
        compose.FuncTransformer,
        compose.Pipeline,
        compose.Grouper,
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
        ignored = (*ignored, compat.PyTorch2CremeRegressor)
    except AttributeError:
        pass

    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, base.Estimator)

    for submodule in importlib.import_module('creme').__all__:

        if submodule == 'base':
            continue

        if submodule == 'synth':
            submodule = 'datasets.synth'

        for _, obj in inspect.getmembers(importlib.import_module(f'creme.{submodule}'), is_estimator):
            if issubclass(obj, ignored):
                continue
            try:
                params = obj._default_params()
            except AttributeError:
                params = {}
            yield obj(**params)


@pytest.mark.parametrize('estimator, check', [
    pytest.param(
        estimator,
        check,
        id=f'{estimator}:{check.__name__}'
    )
    for estimator in list(get_all_estimators()) + [
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
        feature_selection.SelectKBest(similarity=stats.PearsonCorr())
    ]
    for check in utils.estimator_checks.yield_checks(estimator)
])
def test_check_estimator(estimator, check):
    check(copy.deepcopy(estimator))
