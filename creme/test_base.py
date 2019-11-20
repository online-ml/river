from creme import linear_model


def test_reset():

    obj = linear_model.LinearRegression(l2=42)
    obj.fit_one({'x': 3}, 6)

    new = obj._reset(l2=21)
    assert new.l2 == 21
    assert obj.l2 == 42
    assert new.weights == {}
    assert new.weights != obj.weights
