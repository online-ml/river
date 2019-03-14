import pytest

from sklearn.utils import estimator_checks

from creme import cluster
from creme import compat
from creme import linear_model
from creme import preprocessing


@pytest.mark.parametrize(
    'estimator',
    [
        pytest.param(estimator, id=estimator.__class__.__name__)
        for estimator in [
            linear_model.LinearRegression(),
            preprocessing.StandardScaler(),
            preprocessing.OneHotEncoder(),
            cluster.KMeans()
        ]
    ]
)
def test_check_estimator(estimator):
    estimator_checks.check_estimator(compat.creme_to_sklearn(estimator))
