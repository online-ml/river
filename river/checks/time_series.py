from __future__ import annotations

import copy


def check_learn_one(model, dataset):
    """learn_one should accept a target followed by optional exogenous features."""

    for x, y in dataset:
        xx, yy = copy.deepcopy(x), copy.deepcopy(y)

        model.learn_one(y, x)

        assert x == xx
        assert y == yy


def check_forecast(model, dataset):
    """forecast should return one prediction for each requested horizon step."""

    horizon = 3
    xs = [
        {"time": 100.0, "period": 0.0},
        {"time": 101.0, "period": 1.0},
        {"time": 102.0, "period": 2.0},
    ]

    for x, y in dataset:
        model.learn_one(y, x)

    y_pred = model.forecast(horizon=horizon, xs=xs)

    assert isinstance(y_pred, list)
    assert len(y_pred) == horizon
