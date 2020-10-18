from river import linear_model
from river import preprocessing


def test_set_params():

    obj = linear_model.LinearRegression(l2=42)
    obj.learn_one({'x': 3}, 6)

    new = obj._set_params({'l2': 21})
    assert new.l2 == 21
    assert obj.l2 == 42
    assert new.weights == {}
    assert new.weights != obj.weights


def test_set_params_pipeline():

    obj = preprocessing.StandardScaler() | linear_model.LinearRegression(l2=42)
    obj.learn_one({'x': 3}, 6)

    params = {'LinearRegression': {'l2': 21}}
    new = obj._set_params(params)
    assert new['LinearRegression'].l2 == 21
    assert obj['LinearRegression'].l2 == 42
    assert new['LinearRegression'].weights == {}
    assert new['LinearRegression'].weights != obj['LinearRegression'].weights
