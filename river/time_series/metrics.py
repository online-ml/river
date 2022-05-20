import abc
from numbers import Number
from typing import List

from river import base, metrics


class ForecastingMetric(base.Base, abc.ABC):
    @abc.abstractmethod
    def update(self, y_true: List[Number], y_pred: List[Number]) -> "ForecastingMetric":
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
    def get(self) -> List[float]:
        """Return the current performance along the horizon.

        Returns
        -------
        The current performance.

        """


class HorizonMetric(ForecastingMetric):
    """Measures performance at each time step ahead.

    This allows to measure the performance of a model at each time step along the horizon. A copy
    of the provided regression metric is made for each time step.

    Parameters
    ----------
    metric
        A regression metric.

    """

    def __init__(self, metric: metrics.base.RegressionMetric):
        self.metric = metric
        self.metrics = []

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
            f"{prefix:<{prefix_pad}} {metric}"
            for prefix, metric in zip(prefixes, self.metrics)
        )
