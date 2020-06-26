import pytest
from sklearn.utils import estimator_checks

from creme import base
from creme import cluster
from creme import compat
from creme import linear_model
from creme import preprocessing
from sklearn import linear_model as sk_linear_model


@pytest.mark.parametrize('estimator', [
    pytest.param(estimator, id=str(estimator))
    for estimator in [
        linear_model.LinearRegression(),
        preprocessing.StandardScaler(),
        cluster.KMeans(seed=42)
    ]
])
def test_sklearn_check_estimator(estimator: base.Estimator):
    skl_estimator = compat.convert_creme_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)
