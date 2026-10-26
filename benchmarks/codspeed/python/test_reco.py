from marks import benchmark
from workloads import user_item_stream

from river import reco


@benchmark("reco")
def test_biased_mf_learn(benchmark) -> None:
    stream = user_item_stream()

    def run() -> None:
        model = reco.BiasedMF(n_factors=10, seed=42)
        for x, y in stream:
            model.learn_one(**x, y=y)

    benchmark(run)


@benchmark("reco")
def test_funk_mf_learn(benchmark) -> None:
    stream = user_item_stream()

    def run() -> None:
        model = reco.FunkMF(n_factors=10, seed=42)
        for x, y in stream:
            model.learn_one(**x, y=y)

    benchmark(run)
