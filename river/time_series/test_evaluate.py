from river import datasets, metrics, stats, time_series
from river.time_series.evaluate import _evaluate


class MeanForecaster(time_series.Forecaster):
    def __init__(self):
        self.mean = stats.Mean()

    def learn_one(self, y, x=None):
        self.mean.update(y)
        return self

    def forecast(self, horizon, xs=None):
        return [self.mean.get()] * horizon


def test_forecasts_at_each_step():

    dataset = datasets.AirlinePassengers()
    model = MeanForecaster()
    metric = metrics.MAE()
    horizon = 12
    grace_period = 1

    steps = _evaluate(dataset, model, metric, horizon, grace_period)

    y_pred, _ = next(steps)
    assert y_pred == [112] * horizon
    y_pred, _ = next(steps)
    assert y_pred == [(112 + 118) / 2] * horizon
    y_pred, _ = next(steps)
    assert y_pred == [(112 + 118 + 132) / 3] * horizon
    y_pred, _ = next(steps)
    assert y_pred == [(112 + 118 + 132 + 129) / 4] * horizon

    n_steps = sum(1 for _ in _evaluate(dataset, model, metric, horizon, grace_period))
    assert n_steps == dataset.n_samples - horizon - grace_period
