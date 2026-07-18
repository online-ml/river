from __future__ import annotations

import pytest
from workloads import scalar_series

from river import stats

pytestmark = pytest.mark.benchmark(group="stats")


def test_quantile_update(benchmark) -> None:
    series = scalar_series()

    def run() -> float:
        q = stats.Quantile(0.5)
        for x in series:
            q.update(x)
        return q.get()

    benchmark(run)
