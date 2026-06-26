from __future__ import annotations

import functools

import numpy as np
import pytest

from river import utils
from river.neighbors import LazySearch
from river.utils.vectordict import euclidean_distance_tuple


def _fast_path_engine(window_size=100):
    """A `LazySearch` wired to the Rust Euclidean fast path, exactly as KNN/LOF wire it."""
    return LazySearch(window_size=window_size, dist_func=euclidean_distance_tuple)


def _brute_force(query, points, k):
    """Indices of the k nearest points to `query`, sorted by (distance, insertion order)."""
    keys = list(query)
    dists = [sum((query[c] - p[c]) ** 2 for c in keys) ** 0.5 for p in points]
    order = sorted(range(len(points)), key=lambda i: (dists[i], i))[:k]
    return order, [dists[i] for i in order]


@pytest.mark.parametrize("n_neighbors", [1, 3, 10, 50])
def test_euclidean_fast_path_returns_true_knn(n_neighbors):
    """The Rust Euclidean fast path must return the exact k nearest neighbors.

    Regression test: the fast path used to key its heap on the negated distance, which kept
    the *farthest* candidates instead of the nearest.
    """
    rng = np.random.RandomState(0)
    points = [{"a": float(a), "b": float(b)} for a, b in rng.rand(40, 2)]
    query = {"a": 0.5, "b": 0.5}

    engine = _fast_path_engine()
    assert engine.dist_func is euclidean_distance_tuple
    for p in points:
        engine.append((p,))

    _, distances = engine.search((query,), n_neighbors=n_neighbors)

    _, expected = _brute_force(query, points, n_neighbors)
    assert distances == pytest.approx(expected)
    assert all(d1 <= d2 for d1, d2 in zip(distances, distances[1:]))  # sorted ascending


def test_fast_path_matches_python_fallback():
    """The Rust fast path and the pure-Python fallback must agree neighbor-for-neighbor."""
    rng = np.random.RandomState(1)
    points = [{"a": float(a), "b": float(b)} for a, b in rng.rand(60, 2)]
    query = {"a": 0.3, "b": 0.7}

    fast = _fast_path_engine()
    minkowski = functools.partial(utils.math.minkowski_distance, p=2)
    slow = LazySearch(window_size=100, dist_func=lambda a, b: minkowski(a[0], b[0]))
    for p in points:
        fast.append((p,))
        slow.append((p,))

    _, fast_d = fast.search((query,), n_neighbors=15)
    _, slow_d = slow.search((query,), n_neighbors=15)
    assert fast_d == pytest.approx(slow_d)


def test_fewer_points_than_neighbors():
    """Requesting more neighbors than are stored returns all of them, sorted."""
    engine = _fast_path_engine()
    points = [{"a": 0.0}, {"a": 2.0}, {"a": 1.0}]
    for p in points:
        engine.append((p,))
    _, distances = engine.search(({"a": 0.0},), n_neighbors=10)
    assert distances == pytest.approx([0.0, 1.0, 2.0])
