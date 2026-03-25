from __future__ import annotations

import pytest

from river import datasets, evaluate, metrics, stats, time_series


class MeanForecaster(time_series.base.Forecaster):
    def __init__(self):
        self.mean = stats.Mean()

    def learn_one(self, y, x=None):
        self.mean.update(y)

    def forecast(self, horizon, xs=None):
        return [self.mean.get()] * horizon


def test_forecasts_at_each_step():
    dataset = datasets.AirlinePassengers()
    model = MeanForecaster()
    metric = metrics.MAE()
    horizon = 12
    grace_period = 1

    steps = evaluate.iter_evaluate(
        dataset=dataset, model=model, metric=metric, horizon=horizon, grace_period=grace_period
    )

    _, _, y_pred, _ = next(steps)
    assert y_pred == [112] * horizon
    _, _, y_pred, _ = next(steps)
    assert y_pred == [(112 + 118) / 2] * horizon
    _, _, y_pred, _ = next(steps)
    assert y_pred == [(112 + 118 + 132) / 3] * horizon
    _, _, y_pred, _ = next(steps)
    assert y_pred == [(112 + 118 + 132 + 129) / 4] * horizon

    n_steps = sum(
        1
        for _ in evaluate.iter_evaluate(
            dataset=dataset, model=model, metric=metric, horizon=horizon, grace_period=grace_period
        )
    )
    assert n_steps == dataset.n_samples - horizon - grace_period


def test_time_series_evaluate_deprecated():
    dataset = datasets.AirlinePassengers().take(20)
    model = MeanForecaster()
    metric = metrics.MAE()

    with pytest.warns(DeprecationWarning, match="time_series.evaluate"):
        time_series.evaluate(dataset=dataset, model=model, metric=metric, horizon=4)
