from __future__ import annotations

import pytest
from workloads import N_PREDICT, binary_stream

from river import linear_model, optim, preprocessing

pytestmark = pytest.mark.benchmark(group="linear_model")


def test_logistic_regression_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
            optimizer=optim.SGD(0.005)
        )
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


def test_logistic_regression_predict(benchmark) -> None:
    stream = binary_stream()
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression(
        optimizer=optim.SGD(0.005)
    )
    for x, y in stream:
        model.learn_one(x, y)
    xs = [x for x, _ in stream[:N_PREDICT]]

    def run() -> None:
        for x in xs:
            model.predict_proba_one(x)

    benchmark(run)
