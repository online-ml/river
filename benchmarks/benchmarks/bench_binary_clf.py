from .common import BINARY_CLF_DATASETS, make_benchmark_classes
from river import metrics

_classes = make_benchmark_classes(
    track="binary_clf",
    datasets=BINARY_CLF_DATASETS,
    metric_fn=lambda: metrics.Accuracy() + metrics.F1(),
)
globals().update(_classes)
