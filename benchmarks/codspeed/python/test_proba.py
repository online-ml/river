from marks import benchmark
from workloads import regression_stream, scalar_series

from river import proba


@benchmark("proba")
def test_gaussian_update(benchmark) -> None:
    series = scalar_series(5_000)

    def run() -> None:
        model = proba.Gaussian()
        for x in series:
            model.update(x)

    benchmark(run)


@benchmark("proba")
def test_multivariate_gaussian_update(benchmark) -> None:
    stream = [x for x, _ in regression_stream()]

    def run() -> None:
        model = proba.MultivariateGaussian(seed=42)
        for x in stream:
            model.update(x)

    benchmark(run)
