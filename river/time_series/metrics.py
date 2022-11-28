from __future__ import annotations

import abc
import typing
from numbers import Number

from river import base, metrics


class ForecastingMetric(base.Base, abc.ABC):
    @abc.abstractmethod
    def update(self, y_true: list[Number], y_pred: list[Number]) -> ForecastingMetric:
        """Update the metric at each step along the horizon.

        Parameters
        ----------
        y_true
            Ground truth values at each time step of the horizon.
        y_pred
            Predicted values at each time step of the horizon.

        Returns
        -------
        self

        """

    @abc.abstractmethod
    def get(self) -> list[float]:
        """Return the current performance along the horizon.

        Returns
        -------
        The current performance.

        """


class HorizonMetric(ForecastingMetric):
    """Measures performance at each time step ahead.

    This allows to measure the performance of a model at each time step along the horizon. A copy
    of the provided regression metric is made for each time step. At each time step ahead, the
    metric is thus evaluated on each prediction for said time step, and not for the time steps before
    or after that.

    Parameters
    ----------
    metric
        A regression metric.

    Examples
    --------

    This is used internally by the `time_series.evaluate` function.

    >>> from river import datasets
    >>> from river import metrics
    >>> from river import time_series

    >>> metric = time_series.evaluate(
    ...     dataset=datasets.AirlinePassengers(),
    ...     model=time_series.HoltWinters(alpha=0.1),
    ...     metric=metrics.MAE(),
    ...     horizon=4
    ... )

    >>> metric
    +1 MAE: 40.931286
    +2 MAE: 42.667998
    +3 MAE: 44.158092
    +4 MAE: 43.849617

    """

    def __init__(self, metric: metrics.base.RegressionMetric):
        self.metric = metric
        self.metrics: list[metrics.base.RegressionMetric] = []

    def update(self, y_true, y_pred):
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            try:
                metric = self.metrics[t]
            except IndexError:
                metric = self.metric.clone()
                self.metrics.append(metric)

            metric.update(yt, yp)

        return self

    def get(self):
        return [metric.get() for metric in self.metrics]

    def __repr__(self):
        prefixes = [f"+{t+1}" for t in range(len(self.metrics))]
        prefix_pad = max(map(len, prefixes))
        return "\n".join(
            f"{prefix:<{prefix_pad}} {metric}" for prefix, metric in zip(prefixes, self.metrics)
        )


class HorizonAggMetric(HorizonMetric):
    """Same as `HorizonMetric`, but aggregates the result based on an provided function.

    This allows, for instance, to measure the average performance of a forecasting model along the
    horizon.

    Parameters
    ----------
    metric
        A regression metric.
    agg_func
        A function that takes as input a list of floats and outputs a single float. You may want to
        `min`, `max`, as well as `statistics.mean` and `statistics.median`.

    Examples
    --------

    This is used internally by the `time_series.evaluate` function when you pass an `agg_func`.

    >>> import statistics
    >>> from river import datasets
    >>> from river import metrics
    >>> from river import time_series

    >>> metric = time_series.evaluate(
    ...     dataset=datasets.AirlinePassengers(),
    ...     model=time_series.HoltWinters(alpha=0.1),
    ...     metric=metrics.MAE(),
    ...     agg_func=statistics.mean,
    ...     horizon=4
    ... )

    >>> metric
    mean(MAE): 42.901748

    """

    def __init__(
        self, metric: metrics.base.RegressionMetric, agg_func: typing.Callable[[list[float]], float]
    ):
        super().__init__(metric)
        self.agg_func = agg_func

    def get(self):
        return self.agg_func(super().get())

    def __repr__(self):
        name = f"{self.agg_func.__name__}({self.metric.__class__.__name__})"
        return f"{name}: {self.get():{self.metric._fmt}}".rstrip("0")
