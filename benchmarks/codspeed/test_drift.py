from marks import benchmark
from workloads import scalar_series

from river import drift


@benchmark("drift")
def test_adwin_update(benchmark) -> None:
    series = scalar_series()

    def run() -> None:
        detector = drift.ADWIN()
        for x in series:
            detector.update(x)

    benchmark(run)


@benchmark("drift")
def test_kswin_update(benchmark) -> None:
    series = scalar_series(5_000)

    def run() -> None:
        detector = drift.KSWIN(seed=42)
        for x in series:
            detector.update(x)

    benchmark(run)


@benchmark("drift")
def test_page_hinkley_update(benchmark) -> None:
    series = scalar_series()

    def run() -> None:
        detector = drift.PageHinkley()
        for x in series:
            detector.update(x)

    benchmark(run)
