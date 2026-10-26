from marks import benchmark
from workloads import scalar_series

from river import stats


@benchmark("stats")
def test_quantile_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        q = stats.Quantile(0.5)
        for x in series:
            q.update(x)
        return q.get()

    benchmark(run)


@benchmark("stats")
def test_mean_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.Mean()
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_var_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.Var()
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_ewmean_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.EWMean(0.5)
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_skew_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.Skew()
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_kurtosis_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.Kurtosis()
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_rolling_quantile_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.RollingQuantile(q=0.5, window_size=100)
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)


@benchmark("stats")
def test_rolling_iqr_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        model = stats.RollingIQR(window_size=100)
        for x in series:
            model.update(x)
        return model.get()

    benchmark(run)
