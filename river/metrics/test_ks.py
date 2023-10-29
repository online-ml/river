from __future__ import annotations

import numpy as np
import river

from collections import deque
from scipy.stats import ks_2samp


def test_incremental_ks_statistics():
    initial_a = np.random.normal(loc=0, scale=1, size=5)
    initial_b = np.random.normal(loc=1, scale=1, size=5)

    stream_a = np.random.normal(loc=0, scale=1, size=10)
    stream_b = np.random.normal(loc=1, scale=1, size=10)

    incremental_ks_statistics = []
    incremental_ks = river.metrics.KolmogorovSmirnov(statistic="ks")
    sliding_a = deque(initial_a)
    sliding_b = deque(initial_b)

    for a, b in zip(initial_a, initial_b):
        incremental_ks.update(a, 0)
        incremental_ks.update(b, 1)
    for a, b in zip(stream_a, stream_b):
        incremental_ks.revert(sliding_a.popleft(), 0)
        incremental_ks.revert(sliding_b.popleft(), 1)
        sliding_a.append(a)
        sliding_b.append(b)
        incremental_ks.update(a, 0)
        incremental_ks.update(b, 1)
        incremental_ks_statistics.append(incremental_ks.get())

    ks_2samp_statistics = []
    sliding_a = deque(initial_a)
    sliding_b = deque(initial_b)

    for a, b in zip(stream_a, stream_b):
        sliding_a.popleft()
        sliding_b.popleft()
        sliding_a.append(a)
        sliding_b.append(b)
        ks_2samp_statistics.append(ks_2samp(sliding_a, sliding_b).statistic)

    assert np.all(np.isclose(np.array(incremental_ks_statistics), np.array(ks_2samp_statistics)))
