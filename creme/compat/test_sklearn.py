import pytest
from sklearn.utils import estimator_checks

from creme import cluster
from creme import compat
from creme import linear_model
from creme import preprocessing


@pytest.mark.parametrize('estimator', [
    pytest.param(estimator, id=str(estimator))
    for estimator in [
        linear_model.LinearRegression(),
        preprocessing.StandardScaler(),
        cluster.KMeans()
    ]
])
def test_sklearn_check_estimator(estimator):
    skl_estimator = compat.convert_creme_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)
