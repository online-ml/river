from __future__ import annotations

import collections
import numbers
import typing

from river import base, metrics, time_series

TimeSeries = typing.Iterator[
    typing.Tuple[  # noqa: UP006
        typing.Union[dict, None],  # noqa: UP007
        numbers.Number,
        typing.Union[typing.List[dict], None],  # noqa: UP006, UP007
        typing.List[numbers.Number],  # noqa: UP006
    ]
]


def _iter_with_horizon(dataset: base.typing.Dataset, horizon: int) -> TimeSeries:
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

    x_horizon: collections.deque[dict] = collections.deque(maxlen=horizon)
    y_horizon: collections.deque = collections.deque(maxlen=horizon)

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
        yield x_now, y_now, x_horizon, y_horizon  # type: ignore


def iter_evaluate(
    dataset: base.typing.Dataset,
    model: time_series.base.Forecaster,
    metric: metrics.base.RegressionMetric,
    horizon: int,
    agg_func: typing.Callable[[list[float]], float] | None = None,
    grace_period: int | None = None,
):
    """Evaluates the performance of a forecaster on a time series dataset and yields results.

    This does exactly the same as `evaluate.progressive_val_score`. The only difference is that
    this function returns an iterator, yielding results at every step. This can be useful if you
    want to have control over what you do with the results. For instance, you might want to plot
    the results.

    Parameters
    ----------
    dataset
        A sequential time series.
    model
        A forecaster.
    metric
        A regression metric.
    horizon
    agg_func
    grace_period
        Initial period during which the metric is not updated. This is to fairly evaluate models
        which need a warming up period to start producing meaningful forecasts. The value of this
        parameter is equal to the horizon by default.

    """

    horizon_metric = (
        time_series.HorizonAggMetric(metric, agg_func)
        if agg_func
        else time_series.HorizonMetric(metric)
    )
    steps = _iter_with_horizon(dataset, horizon)

    grace_period = horizon if grace_period is None else grace_period
    for _ in range(grace_period):
        x, y, x_horizon, y_horizon = next(steps)
        model.learn_one(y=y, x=x)  # type: ignore

    for x, y, x_horizon, y_horizon in steps:
        y_pred = model.forecast(horizon, xs=x_horizon)
        horizon_metric.update(y_horizon, y_pred)
        model.learn_one(y=y, x=x)  # type: ignore
        yield x, y, y_pred, horizon_metric


def evaluate(
    dataset: base.typing.Dataset,
    model: time_series.base.Forecaster,
    metric: metrics.base.RegressionMetric,
    horizon: int,
    agg_func: typing.Callable[[list[float]], float] | None = None,
    grace_period: int | None = None,
) -> time_series.HorizonMetric:
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
    agg_func
    grace_period
        Initial period during which the metric is not updated. This is to fairly evaluate models
        which need a warming up period to start producing meaningful forecasts. The value of this
        parameter is equal to the horizon by default.

    """

    horizon_metric = None
    steps = iter_evaluate(dataset, model, metric, horizon, agg_func, grace_period)
    for *_, horizon_metric in steps:
        pass

    return horizon_metric
