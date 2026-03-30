from .common import MULTICLASS_CLF_DATASETS, make_benchmark_classes
from river import metrics

_classes = make_benchmark_classes(
    track="multiclass_clf",
    datasets=MULTICLASS_CLF_DATASETS,
    metric_fn=lambda: metrics.Accuracy() + metrics.MicroF1() + metrics.MacroF1(),
)
globals().update(_classes)
