import pytest

from sklearn.utils import estimator_checks

from creme import compat
from creme import linear_model
from creme import preprocessing


@pytest.mark.parametrize(
    'estimator',
    [
        linear_model.LinearRegression(),
        preprocessing.StandardScaler(),
        preprocessing.OneHotEncoder()
    ]
)
def test_check_estimator(estimator):
    estimator_checks.check_estimator(compat.creme_to_sklearn(estimator))
