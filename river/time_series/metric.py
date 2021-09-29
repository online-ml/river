from numbers import Number
from typing import List
from river.metrics import RegressionMetric


class HorizonMetric:
    def __init__(self, metric: RegressionMetric):
        self.metric = metric
        self.metric_at_each_step = []

    def update(self, y_true: List[Number], y_pred: List[Number]) -> "MetricHorizon":
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            try:
                metric = self.metric_at_each_step[t]
            except IndexError:
                metric = self.metric.clone()
                self.metric_at_each_step.append(metric)

            metric.update(yt, yp)

        return self

    def get(self) -> List[float]:
        return [metric.get() for metric in self.metric_at_each_step]
