from __future__ import annotations

import math

import pytest
from sklearn import linear_model as sklm

from river import anomaly, datasets, optim

tests = {
    "Vanilla": (
        {"optimizer": optim.SGD(1e-2), "nu": 0.5},
        {"learning_rate": "constant", "eta0": 1e-2, "nu": 0.5},
    ),
    "No intercept": (
        {"optimizer": optim.SGD(1e-2), "nu": 0.5, "intercept_lr": 0.0},
        {"learning_rate": "constant", "eta0": 1e-2, "nu": 0.5, "fit_intercept": False},
    ),
}


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    tests.values(),
    ids=tests.keys(),
)
def test_sklearn_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results."""

    rv = anomaly.OneClassSVM(**river_params)
    sk = sklm.SGDOneClassSVM(**sklearn_params)

    for x, _ in datasets.Phishing().take(100):
        rv.learn_one(x)
        sk.partial_fit([list(x.values())])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])
