from marks import benchmark
from workloads import user_item_stream

from river import facto


@benchmark("facto")
def test_fm_classifier_learn(benchmark) -> None:
    stream = user_item_stream()

    def run() -> None:
        model = facto.FMClassifier(n_factors=10, seed=42)
        for x, y in stream:
            model.learn_one(x, y > 3)

    benchmark(run)


@benchmark("facto")
def test_ffm_regressor_learn(benchmark) -> None:
    stream = user_item_stream()

    def run() -> None:
        model = facto.FFMRegressor(n_factors=5, seed=42)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
