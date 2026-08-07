from __future__ import annotations

import math
import random

import pytest

from river import linear_model, optim


@pytest.mark.parametrize("quantile", [-0.1, 0.0, 1.0, 1.1])
def test_invalid_quantile(quantile):
    with pytest.raises(ValueError, match="quantile must be strictly between 0 and 1"):
        linear_model.QuantileRegressor(quantile=quantile)


@pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
def test_learns_known_quantile(quantile):
    n = 5_000
    ys = [-math.log(1 - ((i + 0.5) / n)) for i in range(n)]
    random.Random(42).shuffle(ys)

    model = linear_model.QuantileRegressor(
        quantile=quantile,
        optimizer=optim.SGD(0.0),
        intercept_lr=optim.schedulers.InverseScaling(0.2, power=0.51),
    )

    for _ in range(3):
        for y in ys:
            model.learn_one({}, y)

    assert model.quantile == quantile
    assert isinstance(model.loss, optim.losses.Quantile)
    assert model.loss.alpha == quantile
    assert model.predict_one({}) == pytest.approx(-math.log(1 - quantile), abs=0.05)
