from sklearn.utils import estimator_checks

from creme import compat
from creme import linear_model


def test_linear_regression():
    estimator = compat.SKLRegressorWrapper(linear_model.LinearRegression())
    estimator_checks.check_estimator(estimator)
