import collections
import numbers
from typing import Iterable, Iterator, Optional, Tuple

from river.base.typing import Dataset
from river.metrics import RegressionMetric

from .base import Forecaster
from .metrics import HorizonMetric

TimeSeries = Iterator[
    Tuple[
        Optional[dict],  # x
        numbers.Number,  # y
        Iterable[Optional[dict]],  # x_horizon
        Iterable[numbers.Number],  # y_horizon
    ]
]


def _iter_with_horizon(dataset: Dataset, horizon: int) -> TimeSeries:
    """

    Examples
    --------

    >>> from river import datasets
    >>> from river.time_series.evaluate import _iter_with_horizon

    >>> dataset = datasets.AirlinePassengers()

    >>> for x, y, x_horizon, y_horizon in _iter_with_horizon(dataset.take(8), horizon=3):
    ...     print(x['month'].strftime('%Y-%m-%d'), y)
    ...     print([x['month'].strftime('%Y-%m-%d') for x in x_horizon])
    ...     print(list(y_horizon))
    ...     print('---')
    1949-01-01 112
    ['1949-02-01', '1949-03-01', '1949-04-01']
    [118, 132, 129]
    ---
    1949-02-01 118
    ['1949-03-01', '1949-04-01', '1949-05-01']
    [132, 129, 121]
    ---
    1949-03-01 132
    ['1949-04-01', '1949-05-01', '1949-06-01']
    [129, 121, 135]
    ---
    1949-04-01 129
    ['1949-05-01', '1949-06-01', '1949-07-01']
    [121, 135, 148]
    ---
    1949-05-01 121
    ['1949-06-01', '1949-07-01', '1949-08-01']
    [135, 148, 148]
    ---

    """

    x_horizon = collections.deque(maxlen=horizon)
    y_horizon = collections.deque(maxlen=horizon)

    stream = iter(dataset)

    for _ in range(horizon):
        x, y = next(stream)
        x_horizon.append(x)
        y_horizon.append(y)

    for x, y in stream:
        x_now = x_horizon.popleft()
        y_now = y_horizon.popleft()
        x_horizon.append(x)
        y_horizon.append(y)
        yield x_now, y_now, x_horizon, y_horizon


def _evaluate(
    dataset: Dataset,
    model: Forecaster,
    metric: RegressionMetric,
    horizon: int,
    grace_period: int,
) -> HorizonMetric:

    horizon_metric = HorizonMetric(metric)
    steps = _iter_with_horizon(dataset, horizon)

    for _ in range(grace_period):
        x, y, x_horizon, y_horizon = next(steps)
        model.learn_one(y=y, x=x)

    for x, y, x_horizon, y_horizon in steps:
        y_pred = model.forecast(horizon, xs=x_horizon)
        horizon_metric.update(y_horizon, y_pred)
        model.learn_one(y=y, x=x)
        yield y_pred, horizon_metric


def evaluate(
    dataset: Dataset,
    model: Forecaster,
    metric: RegressionMetric,
    horizon: int,
    grace_period=1,
) -> HorizonMetric:
    """Evaluates the performance of a forecaster on a time series dataset.

    To understand why this method is useful, it's important to understand the difference between
    nowcasting and forecasting. Nowcasting is about predicting a value at the next time step. This
    can be seen as a special case of regression, where the value to predict is the value at the
    next time step. In this case, the `evaluate.progressive_val_score` function may be used to
    evaluate a model via progressive validation.

    Forecasting models can also be evaluated via progressive validation. This is the purpose of
    this function. At each time step `t`, the forecaster is asked to predict the values at `t + 1`,
    `t + 2`, ..., `t + horizon`. The performance at each time step is measured and returned.

    Parameters
    ----------
    dataset
        A sequential time series.
    model
        A forecaster.
    metric
        A regression metric.
    horizon
    grace_period
        Initial period during which the metric is not updated. This is to fairly evaluate models
        which need a warming up period to start producing meaningful forecasts. The first forecast
        is skipped by default.

    """

    horizon_metric = None
    steps = _evaluate(dataset, model, metric, horizon, grace_period)
    for _, horizon_metric in steps:
        pass

    return horizon_metric
