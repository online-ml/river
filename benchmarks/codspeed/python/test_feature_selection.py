from marks import benchmark
from workloads import regression_stream

from river import feature_selection, stats


@benchmark("feature_selection")
def test_select_k_best_learn(benchmark) -> None:
    stream = regression_stream()

    def run() -> None:
        model = feature_selection.SelectKBest(similarity=stats.PearsonCorr(), k=5)
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
