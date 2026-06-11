from .common import REGRESSION_DATASETS, make_benchmark_classes
from river import metrics

_classes = make_benchmark_classes(
    track="regression",
    datasets=REGRESSION_DATASETS,
    metric_fn=lambda: metrics.MAE() + metrics.RMSE() + metrics.R2(),
)
globals().update(_classes)
