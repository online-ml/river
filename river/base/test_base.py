from river import datasets
from river import linear_model
from river import optim
from river import preprocessing


def test_set_params():

    obj = linear_model.LinearRegression(l2=42)
    obj.learn_one({"x": 3}, 6)

    new = obj._set_params({"l2": 21})
    assert new.l2 == 21
    assert obj.l2 == 42
    assert new.weights == {}
    assert new.weights != obj.weights


def test_set_params_pipeline():

    obj = preprocessing.StandardScaler() | linear_model.LinearRegression(l2=42)
    obj.learn_one({"x": 3}, 6)

    params = {"LinearRegression": {"l2": 21}}
    new = obj._set_params(params)
    assert new["LinearRegression"].l2 == 21
    assert obj["LinearRegression"].l2 == 42
    assert new["LinearRegression"].weights == {}
    assert new["LinearRegression"].weights != obj["LinearRegression"].weights


def test_clone_idempotent():

    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.Adam(), l2=0.1
    )

    trace = []
    for x, y in datasets.Phishing():
        trace.append(model.predict_proba_one(x))
        model.learn_one(x, y)

    clone = model.clone()
    for i, (x, y) in enumerate(datasets.Phishing()):
        assert clone.predict_proba_one(x) == trace[i]
        clone.learn_one(x, y)
