from marks import heavy
from workloads import binary_stream, scalar_series

from river import drift, tree


@heavy("drift")
def test_adwin_update(benchmark) -> None:
    series = scalar_series()

    def run() -> None:
        detector = drift.ADWIN()
        for x in series:
            detector.update(x)

    benchmark(run)


@heavy("drift")
def test_kswin_update(benchmark) -> None:
    # KSWIN runs a KS test per sample; 1,000 samples still exercise ~900 tests
    # after the window fills, at a fifth of the CPU-simulation cost.
    series = scalar_series(1_000)

    def run() -> None:
        detector = drift.KSWIN(seed=42)
        for x in series:
            detector.update(x)

    benchmark(run)


@heavy("drift")
def test_d3_update(benchmark) -> None:
    series = [x for x, _ in binary_stream(1_000)]

    def run() -> None:
        detector = drift.D3(
            tree.HoeffdingTreeClassifier(grace_period=40, max_depth=3),
            window_size=200,
        )
        for x in series:
            detector.update(x)

    benchmark(run)


@heavy("drift")
def test_page_hinkley_update(benchmark) -> None:
    series = scalar_series()

    def run() -> None:
        detector = drift.PageHinkley()
        for x in series:
            detector.update(x)

    benchmark(run)
