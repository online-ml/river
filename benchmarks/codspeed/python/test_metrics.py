from marks import benchmark
from workloads import label_pairs, multiclass_stream, score_pairs

from river import metrics


@benchmark("metrics")
def test_accuracy_update(benchmark) -> None:
    pairs = label_pairs()

    def run() -> None:
        metric = metrics.Accuracy()
        for y_true, y_pred in pairs:
            metric.update(y_true, y_pred)

    benchmark(run)


@benchmark("metrics")
def test_f1_update(benchmark) -> None:
    pairs = label_pairs()

    def run() -> None:
        metric = metrics.F1()
        for y_true, y_pred in pairs:
            metric.update(y_true, y_pred)

    benchmark(run)


@benchmark("metrics")
def test_rocauc_update(benchmark) -> None:
    pairs = score_pairs()

    def run() -> None:
        metric = metrics.ROCAUC()
        for y_true, score in pairs:
            metric.update(y_true, score)

    benchmark(run)


@benchmark("metrics")
def test_rolling_rocauc_update(benchmark) -> None:
    pairs = score_pairs()

    def run() -> None:
        metric = metrics.RollingROCAUC(window_size=250)
        for y_true, score in pairs:
            metric.update(y_true, score)

    benchmark(run)


@benchmark("metrics")
def test_classification_report_update(benchmark) -> None:
    labels = [y for _, y in multiclass_stream()]
    pairs = list(zip(labels, labels[1:] + labels[:1]))

    def run() -> None:
        metric = metrics.ClassificationReport()
        for y_true, y_pred in pairs:
            metric.update(y_true, y_pred)

    benchmark(run)
