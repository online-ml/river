from sklearn.utils import estimator_checks

from creme import compat
from creme import linear_model
from creme import preprocessing


def test_linear_regression():
    estimator = compat.SKLRegressorWrapper(linear_model.LinearRegression())
    estimator_checks.check_estimator(estimator)


def test_standard_scaler():
    estimator = compat.SKLTransformerWrapper(preprocessing.StandardScaler())
    estimator_checks.check_estimator(estimator)
