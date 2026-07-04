"""Deterministic, network-free workloads shared by all CodSpeed benchmarks.

Every helper is cached and returns the same object to all callers: treat the
returned lists as frozen. All generators are seeded; nothing here touches the
network or the clock.
"""

from __future__ import annotations

import functools
import math

from river.datasets import synth

N_LEARN = 1_000
N_PREDICT = 200


@functools.cache
def binary_stream(n: int = N_LEARN) -> list[tuple[dict[str, float], bool]]:
    """Seeded binary-classification stream with three numeric features."""
    return list(synth.SEA(variant=0, seed=42).take(n))


@functools.cache
def scalar_series(n: int = 10_000) -> list[float]:
    """Deterministic univariate series; mirrors benches/stats_bench.rs exactly."""
    return [math.sin(i * 0.7) for i in range(n)]
