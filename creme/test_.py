"""
General tests for all estimators.
"""
import copy
import importlib
import inspect
import itertools

import pytest

import creme
from creme import base
from creme import dummy
from creme import cluster
from creme import compose
from creme import datasets
from creme import ensemble
from creme import feature_extraction
from creme import feature_selection
from creme import linear_model
from creme import multiclass
from creme import multioutput
from creme import naive_bayes
from creme import optim
from creme import preprocessing
from creme import stats
from creme import stream
from creme import time_series
from creme import tree
from creme import utils
from creme.compat.sklearn import CremeBaseWrapper
from creme.compat.sklearn import SKLBaseWrapper

from sklearn import datasets

def get_all_estimators():

    ignored = (
        CremeBaseWrapper,
        SKLBaseWrapper,
        compose.FuncTransformer,
        compose.TargetModifierRegressor,
        ensemble.StackingBinaryClassifier,
        feature_extraction.Agg,
        feature_extraction.TargetAgg,
        feature_extraction.Differ,
        linear_model.FMRegressor,
        linear_model.SoftmaxRegression,
        multioutput.ClassifierChain,
        multioutput.RegressorChain,
        preprocessing.OneHotEncoder,
        time_series.Detrender,
        time_series.GroupDetrender,
        time_series.SNARIMAX
    )

    def is_estimator(obj):
        return inspect.isclass(obj) and issubclass(obj, base.Estimator)

    for submodule in importlib.import_module('creme').__all__:

        if submodule == 'base':
            continue

        for name, obj in inspect.getmembers(importlib.import_module(f'creme.{submodule}'), is_estimator):

            if issubclass(obj, ignored):
                continue

            elif issubclass(obj, dummy.StatisticRegressor):
                inst = obj(statistic=stats.Mean())

            elif issubclass(obj, compose.BoxCoxTransformRegressor):
                inst = obj(regressor=linear_model.LinearRegression())

            elif issubclass(obj, tree.RandomForestClassifier):
                inst = obj()

            elif issubclass(obj, ensemble.BaggingClassifier):
                inst = obj(linear_model.LogisticRegression())

            elif issubclass(obj, ensemble.BaggingRegressor):
                inst = obj(linear_model.LinearRegression())

            elif issubclass(obj, ensemble.AdaBoostClassifier):
                inst = obj(linear_model.LogisticRegression())

            elif issubclass(obj, ensemble.HedgeRegressor):
                inst = obj([
                    preprocessing.StandardScaler() | linear_model.LinearRegression(intercept_lr=0.1),
                    preprocessing.StandardScaler() | linear_model.PARegressor(),
                ])

            elif issubclass(obj, feature_selection.RandomDiscarder):
                inst = obj(n_to_keep=5)

            elif issubclass(obj, feature_selection.SelectKBest):
                inst = obj(similarity=stats.PearsonCorrelation())

            elif issubclass(obj, linear_model.LinearRegression):
                inst = preprocessing.StandardScaler() | obj(intercept_lr=0.1)

            elif issubclass(obj, linear_model.PARegressor):
                inst = preprocessing.StandardScaler() | obj()

            elif issubclass(obj, multiclass.OneVsRestClassifier):
                inst = obj(binary_classifier=linear_model.LogisticRegression())

            else:
                inst = obj()

            yield inst


@pytest.mark.parametrize(
    'estimator',
    [
        pytest.param(copy.deepcopy(estimator), id=str(estimator))
        for estimator in list(get_all_estimators()) + [
            feature_extraction.TFIDFVectorizer(),
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
            preprocessing.PolynomialExtender(),
            feature_selection.VarianceThreshold(),
            feature_selection.SelectKBest(similarity=stats.PearsonCorrelation())
        ]
    ]
)
def test_check_estimator(estimator):
    utils.estimator_checks.check_estimator(estimator)


@pytest.mark.parametrize(
    'optimizer, initializer',
    itertools.product([
        pytest.param(copy.deepcopy(optimizer), id=str(optimizer))
            for optimizer in [
                optim.AdaBound(),
                optim.AdaDelta(),
                optim.AdaGrad(),
                optim.AdaMax(),
                optim.Adam(),
                optim.Momentum(),
                optim.NesterovMomentum(),
                optim.RMSProp(),
                optim.SGD()
            ]
        ],
        [
            pytest.param(copy.deepcopy(initializer), id=str(initializer))
            for initializer in [
                optim.initializers.Zeros(),
                optim.initializers.Constant(value=3.14),
                optim.initializers.Normal(mu=0, sigma=1, random_state=42)
            ]
        ]
    )
)
def test_check_gradient(optimizer, initializer):
    # Test gradient finite difference with linear regression
    X_y = stream.iter_sklearn_dataset(
        dataset=datasets.load_boston(),
        shuffle=True,
        random_state=42
    )

    scaler = preprocessing.StandardScaler()
    model = linear_model.LinearRegression()
    model.weights = initializer(shape = len(model.weights))

    for x, y in X_y:
        x = scaler.fit_one(x).transform_one(x)
        utils.estimator_checks.check_gradient_finite_difference(model, x, y, delta = 0.001)
        model.fit_one(x, y)
        
    # Test gradient finite difference with logistic regression
    X_y = creme.datasets.fetch_electricity()
    scaler = preprocessing.StandardScaler()
    model = linear_model.LogisticRegression()
    model.weights = initializer(shape = len(model.weights))

    for x, y in X_y:
        x = scaler.fit_one(x).transform_one(x)
        utils.estimator_checks.check_gradient_finite_difference(model, x, y, delta = 0.001)
        model.fit_one(x, y)
