import pytest
from sklearn.utils import estimator_checks
from sklearn import linear_model as sk_linear_model

from river import base
from river import cluster
from river import compat
from river import linear_model
from river import preprocessing


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(estimator, id=str(estimator))
        for estimator in [
            linear_model.LinearRegression(),
            linear_model.LogisticRegression(),
            preprocessing.StandardScaler(),
            cluster.KMeans(seed=42),
        ]
    ],
)
def test_river_to_sklearn_check_estimator(estimator: base.Estimator):
    skl_estimator = compat.convert_river_to_sklearn(estimator)
    estimator_checks.check_estimator(skl_estimator)


def test_sklearn_check_twoway():
    estimator = sk_linear_model.SGDRegressor()
    river_estimator = compat.convert_sklearn_to_river(estimator)
    skl_estimator = compat.convert_river_to_sklearn(river_estimator)
    estimator_checks.check_estimator(skl_estimator)
