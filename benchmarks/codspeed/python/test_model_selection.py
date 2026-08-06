from marks import benchmark
from workloads import regression_stream

from river import linear_model, model_selection


@benchmark("model_selection")
def test_greedy_regressor_learn(benchmark) -> None:
    stream = regression_stream()

    def run() -> None:
        model = model_selection.GreedyRegressor(
            [linear_model.LinearRegression(l2=l2) for l2 in (0.0, 0.01, 0.1)]
        )
        for x, y in stream:
            model.learn_one(x, y)

    benchmark(run)
