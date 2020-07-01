import pytest
from sklearn.utils import estimator_checks
from sklearn import linear_model as sk_linear_model

from creme import base
from creme import cluster
from creme import compat
from creme import linear_model
from creme import preprocessing


@pytest.mark.parametrize('estimator', [
    pytest.param(estimator, id=str(estimator))
    for estimator in [
        linear_model.LinearRegression(),
        linear_model.LogisticRegression(),
        preprocessing.StandardScaler(),
        cluster.KMeans(seed=42)
    ]
])
def test_creme_to_sklearn_check_estimator(estimator: base.Estimator):
    skl_estimator = compat.convert_creme_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)


def test_sklearn_check_twoway():
    estimator = sk_linear_model.SGDRegressor()
    creme_estimator = compat.convert_sklearn_to_creme(estimator)
    skl_estimator = compat.convert_creme_to_sklearn(creme_estimator)
    estimator_checks.check_estimator(skl_estimator)
