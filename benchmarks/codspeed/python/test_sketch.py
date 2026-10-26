from marks import benchmark
from workloads import integer_keys, scalar_series

from river import sketch


@benchmark("sketch")
def test_counter_update(benchmark) -> None:
    keys = integer_keys()

    def run() -> None:
        model = sketch.Counter(seed=42)
        for key in keys:
            model.update(key)

    benchmark(run)


@benchmark("sketch")
def test_heavy_hitters_update(benchmark) -> None:
    keys = integer_keys()

    def run() -> None:
        model = sketch.HeavyHitters()
        for key in keys:
            model.update(key)

    benchmark(run)


@benchmark("sketch")
def test_histogram_update(benchmark) -> None:
    series = scalar_series(5_000)

    def run() -> None:
        model = sketch.Histogram(max_bins=64)
        for x in series:
            model.update(x)

    benchmark(run)
