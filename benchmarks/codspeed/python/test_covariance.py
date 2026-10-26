from marks import benchmark
from workloads import regression_stream

from river import covariance


@benchmark("covariance")
def test_empirical_covariance_update(benchmark) -> None:
    stream = [x for x, _ in regression_stream()]

    def run() -> None:
        model = covariance.EmpiricalCovariance()
        for x in stream:
            model.update(x)

    benchmark(run)
