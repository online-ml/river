from __future__ import annotations

import abc
import typing
from numbers import Number

from river import base, conf

# Using the metrics.py framework
class ForecastingInterval(base.Base, abc.ABC):
    @abc.abstractmethod
    def update(self, y_true: list[Number], y_pred: list[Number]) -> ForecastingInterval:
        """Update the prediction interval along the horizon.

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
        """Return the current prediction interval along the horizon.

        Returns
        -------
        The current performance.

        """


class HorizonInterval(ForecastingInterval):
    """Measures the prediction interval at each time step ahead.

    This allows to measure the performance of a model at each time step along the horizon.

    Parameters
    ----------
    interval
        A regression interval.

    Examples
    --------

    This is used internally by the `time_series.evaluate` function.

    >>> from river import datasets
    >>> from river import metrics
    >>> from river import conf
    >>> from river import time_series

    >>> metric = time_series.evaluate(
    ...     dataset=datasets.AirlinePassengers(),
    ...     model=time_series.HoltWinters(alpha=0.1),
    ...     metric=metrics.MAE(),
    ...     conformal_prediction=conformal_prediction.ICP(),
    ...     horizon=4
    ... )

    >>> metric
    +1 MAE: 40.931286
    +2 MAE: 42.667998
    +3 MAE: 44.158092
    +4 MAE: 43.849617

    """

    def __init__(self, interval: conf.base.Interval):
        self.interval = interval
        self.intervals: list[conf.base.Interval] = []

    def update(self, y_true, y_pred):
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            try:
                interval = self.intervals[t]
            except IndexError:
                interval = self.interval.clone()
                self.intervals.append(interval)

            interval.update(yt, yp)

        return self

    def get(self):
        return [interval.get() for interval in self.intervals]

    def __repr__(self):
        prefixes = [f"+{t+1}" for t in range(len(self.intervals))]
        prefix_pad = max(map(len, prefixes))
        return "\n".join(
            f"{prefix:<{prefix_pad}} {interval.get()}" for prefix, interval in zip(prefixes, self.intervals)
        )
