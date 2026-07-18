from marks import heavy
from workloads import binary_stream

from river import ensemble, tree


@heavy("ensemble")
def test_bagging_learn(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> None:
        model = ensemble.BaggingClassifier(tree.HoeffdingTreeClassifier(), n_models=5, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@heavy("ensemble")
def test_srp_learn(benchmark) -> None:
    stream = binary_stream(300)

    def run() -> None:
        model = ensemble.SRPClassifier(n_models=3, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)


@heavy("ensemble")
def test_adwin_bagging_learn(benchmark) -> None:
    stream = binary_stream(500)

    def run() -> None:
        model = ensemble.ADWINBaggingClassifier(tree.HoeffdingTreeClassifier(), n_models=5, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
