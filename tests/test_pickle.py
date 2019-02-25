import pickle
import pytest

from creme import linear_model


@pytest.mark.parametrize(
    'estimator',
    [linear_model.LinearRegression()]
)
def test_pickling(estimator):
    pickled = pickle.dumps(estimator)
    unpickled = pickle.loads(pickled)
    assert type(unpickled) == type(estimator)
