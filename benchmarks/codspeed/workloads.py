"""Deterministic, network-free workloads shared by all CodSpeed benchmarks.

Every helper is cached and returns the same object to all callers: treat the
returned lists as frozen. All generators are seeded; nothing here touches the
network or the clock.
"""

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


@functools.cache
def multiclass_stream(n: int = N_LEARN) -> list[tuple[dict, str]]:
    """Seeded 10-class stream, 24 binary features (7 informative + 17 irrelevant)."""
    return list(synth.LED(seed=42, irrelevant_features=True).take(n))


@functools.cache
def regression_stream(n: int = N_LEARN) -> list[tuple[dict, float]]:
    """Seeded regression stream (Friedman #1, 10 numeric features)."""
    return list(synth.Friedman(seed=42).take(n))


@functools.cache
def high_dim_stream(n: int = N_LEARN) -> list[tuple[dict, bool]]:
    """Seeded 50-feature binary stream, for models whose cost scales with width."""
    gen = synth.Hyperplane(seed=42, n_features=50, noise_percentage=0.0)
    return list(gen.take(n))


@functools.cache
def text_stream(n: int = 500) -> list[str]:
    """Deterministic pseudo-sentences built from a fixed vocabulary (no RNG)."""
    vocab = [
        "river",
        "stream",
        "online",
        "learning",
        "model",
        "feature",
        "drift",
        "metric",
        "window",
        "rolling",
        "forest",
        "tree",
        "linear",
        "bayes",
        "cluster",
        "anomaly",
        "sketch",
        "quantile",
        "gradient",
        "update",
    ]
    return [" ".join(vocab[(i * 7 + j * 3) % len(vocab)] for j in range(12)) for i in range(n)]


@functools.cache
def user_item_stream(n: int = N_LEARN) -> list[tuple[dict, float]]:
    """Deterministic pseudo-ratings: 50 users x 200 items, ratings in [1, 5]."""
    return [
        ({"user": f"u{i % 50}", "item": f"i{(i * 13) % 200}"}, float(1 + (i * 7919) % 5))
        for i in range(n)
    ]


@functools.cache
def label_pairs(n: int = N_LEARN) -> list[tuple[bool, bool]]:
    """(y_true, y_pred) pairs for classification metrics; fixed error pattern."""
    return [(i % 3 != 0, i % 5 != 0) for i in range(n)]


@functools.cache
def score_pairs(n: int = N_LEARN) -> list[tuple[bool, float]]:
    """(y_true, score) pairs for ranking metrics such as ROCAUC."""
    return [(i % 3 != 0, math.sin(i * 0.7) / 2 + 0.5) for i in range(n)]


@functools.cache
def multioutput_stream(n: int = N_LEARN) -> list[tuple[dict, dict[int, bool]]]:
    """Binary stream with a deterministic 3-label target for multioutput models."""
    return [
        (
            x,
            {
                0: y,
                1: x[0] > 5,
                2: x[1] + x[2] > 10,
            },
        )
        for x, y in binary_stream(n)
    ]


@functools.cache
def categorical_stream(n: int = N_LEARN) -> list[dict[str, str]]:
    """Deterministic categorical stream with five keys and 20 values per key."""
    return [{f"cat_{j}": f"v{(i + j * 7) % 20}" for j in range(5)} for i in range(n)]


@functools.cache
def integer_keys(n: int = 5_000) -> list[int]:
    """Deterministic integer keys for sketch benchmarks."""
    return [(i * 17) % 100 for i in range(n)]


@functools.cache
def seasonal_series(n: int = N_LEARN) -> list[float]:
    """Deterministic seasonal signal for Holt-Winters benchmarks."""
    return [math.sin(2 * math.pi * i / 12) + i / 500 for i in range(n)]
