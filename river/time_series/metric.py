from numbers import Number
from typing import List
from river.metrics import RegressionMetric


class HorizonMetric:
    def __init__(self, metric: RegressionMetric):
        self.metric = metric
        self.metrics = []

    def update(self, y_true: List[Number], y_pred: List[Number]) -> "MetricHorizon":
        for t, (yt, yp) in enumerate(zip(y_true, y_pred)):
            try:
                metric = self.metrics[t]
            except IndexError:
                metric = self.metric.clone()
                self.metrics.append(metric)

            metric.update(yt, yp)

        return self

    def get(self) -> List[float]:
        return [metric.get() for metric in self.metrics]

    def __repr__(self):
        prefixes = [f"+{t}" for t in range(len(self.metrics))]
        prefix_pad = max(map(len, prefixes))
        return "\n".join(
            f"{prefix:<{prefix_pad}} {metric}"
            for prefix, metric in zip(prefixes, self.metrics)
        )
