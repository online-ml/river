import collections
from typing import Any, Iterable, Iterator, List, Optional, Tuple
from river.base.typing import Dataset
from river.metrics import RegressionMetric
from river import base, utils

from .base import Forecaster
from .metric import HorizonMetric

TimeSeries = Iterator[
    Tuple[
        Optional[dict],  # x
        Any,  # y
        Iterable[Optional[dict]],  # x_horizon
        Iterable[Any],  # y_horizon
    ]
]


def _iter_with_horizon(dataset: Dataset, horizon: int) -> TimeSeries:

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
    dataset: Dataset, model: Forecaster, metric: RegressionMetric, horizon: int
) -> HorizonMetric:

    horizon_metric = HorizonMetric(metric)

    for x, y, x_horizon, y_horizon in _iter_with_horizon(dataset, horizon):
        model.learn_one(y=y, x=x)
        y_pred = model.forecast(horizon, xs=x_horizon)
        horizon_metric.update(y_horizon, y_pred)

        yield y_pred, horizon_metric


def evaluate(
    dataset: Dataset, model: Forecaster, metric: RegressionMetric, horizon: int
) -> HorizonMetric:

    steps = _evaluate(dataset, model, metric, horizon)
    for _, horizon_metric in steps:
        pass

    return horizon_metric
