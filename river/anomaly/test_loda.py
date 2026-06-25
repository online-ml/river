from __future__ import annotations

import random

from river import anomaly


def _blob(rng, mean, d=5):
    return {f"f{j}": rng.gauss(mean, 1.0) for j in range(d)}


def test_outliers_score_higher():
    """Samples far from the training distribution get higher scores."""
    rng = random.Random(0)
    loda = anomaly.LODA(seed=42)
    for _ in range(2000):
        loda.learn_one(_blob(rng, 0.0))

    inliers = sum(loda.score_one(_blob(rng, 0.0)) for _ in range(300)) / 300
    outliers = sum(loda.score_one(_blob(rng, 8.0)) for _ in range(300)) / 300
    assert outliers > inliers


def test_reproducible_with_seed():
    """Two detectors with the same seed produce identical scores."""
    rng = random.Random(0)
    data = [_blob(rng, 0.0) for _ in range(500)]

    a = anomaly.LODA(seed=1)
    b = anomaly.LODA(seed=1)
    for x in data:
        a.learn_one(x)
        b.learn_one(x)

    probe = _blob(random.Random(7), 3.0)
    assert a.score_one(probe) == b.score_one(probe)


def test_different_seeds_differ():
    rng = random.Random(0)
    data = [_blob(rng, 0.0) for _ in range(500)]

    a = anomaly.LODA(seed=1)
    b = anomaly.LODA(seed=2)
    for x in data:
        a.learn_one(x)
        b.learn_one(x)

    probe = _blob(random.Random(7), 3.0)
    assert a.score_one(probe) != b.score_one(probe)


def test_n_nonzero_features():
    """Each projection keeps exactly floor(sqrt(d)) non-zero weights."""
    rng = random.Random(0)
    loda = anomaly.LODA(seed=42)
    loda.learn_one(_blob(rng, 0.0, d=16))  # sqrt(16) == 4
    assert loda.n_features == 16
    assert all(len(p) == 4 for p in loda.projections_)


def test_score_before_learn_is_zero():
    assert anomaly.LODA().score_one({"a": 1.0}) == 0.0


def test_missing_feature_treated_as_zero():
    """A missing feature must not raise; it is treated as a zero."""
    rng = random.Random(0)
    loda = anomaly.LODA(seed=42)
    for _ in range(200):
        loda.learn_one(_blob(rng, 0.0, d=5))
    # Drop a feature at scoring time: should not raise.
    x = _blob(rng, 0.0, d=5)
    del x["f2"]
    loda.score_one(x)
