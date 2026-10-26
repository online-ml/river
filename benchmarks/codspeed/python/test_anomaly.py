from marks import benchmark
from workloads import binary_stream

from river import anomaly


@benchmark("anomaly")
def test_half_space_trees_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = anomaly.HalfSpaceTrees(seed=42)
        for x, _ in stream:
            model.learn_one(x)

    benchmark(run)


@benchmark("anomaly")
def test_half_space_trees_score(benchmark) -> None:
    stream = binary_stream()
    model = anomaly.HalfSpaceTrees(seed=42)
    for x, _ in stream:
        model.learn_one(x)

    def run() -> None:
        for x, _ in stream:
            model.score_one(x)

    benchmark(run)


@benchmark("anomaly")
def test_one_class_svm_learn(benchmark) -> None:
    stream = binary_stream()

    def run() -> None:
        model = anomaly.OneClassSVM(nu=0.1)
        for x, _ in stream:
            model.learn_one(x)

    benchmark(run)
