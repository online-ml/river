from marks import benchmark
from workloads import scalar_series, seasonal_series

from river import time_series


@benchmark("time_series")
def test_snarimax_learn(benchmark) -> None:
    series = scalar_series(1_000)

    def run() -> None:
        model = time_series.SNARIMAX(p=2, d=0, q=2)
        for y in series:
            model.learn_one(y)

    benchmark(run)


@benchmark("time_series")
def test_holt_winters_learn(benchmark) -> None:
    series = seasonal_series()

    def run() -> None:
        model = time_series.HoltWinters(alpha=0.3, beta=0.1, gamma=0.5, seasonality=12)
        for y in series:
            model.learn_one(y)

    benchmark(run)
