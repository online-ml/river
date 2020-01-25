from sklearn.utils import estimator_checks

from creme import compat
from creme import linear_model


def test_check_estimator():

    model = linear_model.LinearRegression()
    skl_model = compat.convert_creme_to_sklearn(model)
    estimator_checks.check_estimator(skl_model)
