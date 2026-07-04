"""Deterministic workloads for the estimator regression suite.

Every workload is materialized fully into memory before the evaluated loop runs.
None of them touch the network, read the wall clock, or use unseeded randomness.
Workload IDs are versioned (``*_v1``); changing the generated data requires a
new version suffix so scenario history stays comparable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

from river.datasets import synth

N_SAMPLES = 1_000
N_HEAVY_SAMPLES = 300
SEED = 42

Sample = tuple[dict[Any, Any], Any]


def _take(n: int, stream: Iterable[Sample]) -> list[Sample]:
    """Materialize ``n`` samples from ``stream`` into a list.

    Materializing outside the evaluated loop keeps the harness free of any
    generator state and makes repeated runs byte-identical.
    """

    return [sample for _, sample in zip(range(n), stream, strict=False)]


def binary_sea_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Low-dimensional numeric binary classification stream (SEA variant 0)."""

    return _take(n, synth.SEA(variant=0, seed=SEED))


def multiclass_led_v1(n: int = N_SAMPLES) -> list[Sample]:
    """10-class categorical multiclass stream (LED with irrelevant features)."""

    return _take(n, synth.LED(seed=SEED, irrelevant_features=True))


def regression_friedman_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Numeric regression stream (Friedman #1)."""

    return _take(n, synth.Friedman(seed=SEED))


def categorical_5x20_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Count-valued features for count-aware naive Bayes models.

    Five features, each a small non-negative integer count, with a boolean
    target that depends on whether any feature exceeds a fixed threshold.
    """

    rng = random.Random(SEED)
    samples: list[Sample] = []
    for _ in range(n):
        counts = [rng.randint(0, 19) for _ in range(5)]
        x = {f"c{i}": float(c) for i, c in enumerate(counts)}
        y = any(c >= 10 for c in counts)
        samples.append((x, y))
    return samples


def user_item_ratings_v1(n: int = N_SAMPLES) -> list[tuple[int, int, float]]:
    """Deterministic user-item-rating stream for recommenders.

    Returns ``(user, item, rating)`` triples. Ratings are derived from a small
    latent matrix so a recommender has signal to learn.
    """

    rng = random.Random(SEED)
    n_users, n_items = 20, 15
    user_bias = [rng.gauss(0, 0.1) for _ in range(n_users)]
    item_bias = [rng.gauss(0, 0.1) for _ in range(n_items)]
    samples: list[tuple[int, int, float]] = []
    for _ in range(n):
        u = rng.randint(0, n_users - 1)
        i = rng.randint(0, n_items - 1)
        rating = 1.0 + ((u + i) % 5) + user_bias[u] + item_bias[i]
        samples.append((u, i, max(1.0, min(5.0, rating))))
    return samples


def seasonal_series_v1(n: int = N_SAMPLES) -> list[tuple[dict[str, float], float]]:
    """Deterministic seasonal time series for forecasting.

    Returns ``(x, y)`` pairs where ``x`` carries an exogenous ``t`` index and
    ``y`` is a noisy sinusoid with a linear trend. Forecasters consume the
    series in order; ``x`` may be ignored by models that only use the target.
    """

    rng = random.Random(SEED)
    samples: list[tuple[dict[str, float], float]] = []
    for t in range(n):
        trend = 0.01 * t
        seasonal = math.sin(2 * math.pi * t / 12)
        noise = rng.gauss(0, 0.05)
        y = trend + seasonal + noise + 1.0
        samples.append(({"t": float(t)}, y))
    return samples


def cluster_blobs_v1(n: int = N_SAMPLES) -> list[dict[str, float]]:
    """Three fixed Gaussian blobs for clustering (no global RNG dependency)."""

    centers = [(0.0, 0.0), (5.0, 5.0), (-5.0, 5.0)]
    rng = random.Random(SEED)
    samples: list[dict[str, float]] = []
    for i in range(n):
        cx, cy = centers[i % len(centers)]
        x = cx + rng.gauss(0, 0.5)
        y = cy + rng.gauss(0, 0.5)
        samples.append({"x0": x, "x1": y})
    return samples


def anomaly_mixture_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Deterministic normal/anomaly mixture for anomaly detectors.

    Anomalies are ~10% of the stream, drawn from a shifted distribution. The
    boolean label marks true anomalies for evaluation.
    """

    rng = random.Random(SEED)
    samples: list[Sample] = []
    for i in range(n):
        is_anomaly = (i % 10) == 0
        if is_anomaly:
            x = {f"f{j}": rng.gauss(8, 0.5) for j in range(4)}
        else:
            x = {f"f{j}": rng.gauss(0, 1.0) for j in range(4)}
        samples.append((x, is_anomaly))
    return samples


def multioutput_binary_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Multi-label binary classification stream derived from SEA.

    Two labels: the original SEA target and a second label derived from the
    first feature crossing a threshold.
    """

    base = binary_sea_v1(n)
    samples: list[Sample] = []
    for x, y in base:
        label_b = float(x[0]) > 5.0
        samples.append((x, {"a": bool(y), "b": bool(label_b)}))
    return samples


def multitarget_regression_v1(n: int = N_SAMPLES) -> list[Sample]:
    """Multi-target regression stream derived from Friedman.

    Two targets: the Friedman target and a deterministic transform of it.
    """

    base = regression_friedman_v1(n)
    samples: list[Sample] = []
    for x, y in base:
        samples.append((x, {"y0": float(y), "y1": float(y) * 0.5 + 1.0}))
    return samples
