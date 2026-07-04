from marks import benchmark
from workloads import binary_stream

from river import evaluate, linear_model, metrics, preprocessing


@benchmark("evaluate")
def test_progressive_val_score_macro(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> metrics.Accuracy:
        return evaluate.progressive_val_score(
            dataset=stream,
            model=preprocessing.StandardScaler() | linear_model.LogisticRegression(),
            metric=metrics.Accuracy(),
            print_every=0,
            show_time=False,
            show_memory=False,
        )

    benchmark(run)
