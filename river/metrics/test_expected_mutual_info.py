"""Tests for the expected mutual information implementation.

These tests pin down the behaviour we expect from `expected_mutual_info`. They
exist primarily so the Cython implementation can be replaced with a Rust port
without changing observable behaviour.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from sklearn.metrics.cluster import expected_mutual_information as sk_emi

from river import metrics
from river.metrics.expected_mutual_info import expected_mutual_info


def _cm_from_contingency(contingency: np.ndarray) -> metrics.ConfusionMatrix:
    """Build a river ConfusionMatrix matching the given contingency table."""
    cm = metrics.ConfusionMatrix()
    rows, cols = contingency.shape
    for i in range(rows):
        for j in range(cols):
            for _ in range(int(contingency[i, j])):
                cm.update(i, j)
    return cm


@pytest.mark.parametrize(
    "contingency",
    [
        np.array([[5, 1, 0], [0, 3, 2], [1, 0, 4]]),
        np.array([[10, 0], [0, 10]]),
        np.array([[3, 3], [3, 3]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[20, 1, 1, 1], [1, 18, 2, 0], [0, 1, 15, 4], [2, 0, 1, 17]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ],
)
def test_matches_sklearn(contingency):
    cm = _cm_from_contingency(contingency)
    expected = sk_emi(contingency, int(contingency.sum()))
    actual = expected_mutual_info(cm)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_single_row_class_returns_zero():
    """If only one true class is present, EMI is 0."""
    cm = _cm_from_contingency(np.array([[5, 3, 2]]))
    assert expected_mutual_info(cm) == 0.0


def test_single_col_class_returns_zero():
    """If only one predicted class is present, EMI is 0."""
    cm = _cm_from_contingency(np.array([[5], [3], [2]]))
    assert expected_mutual_info(cm) == 0.0


def test_perfect_agreement():
    """Diagonal contingency table — EMI is small but nonzero (and matches sklearn)."""
    contingency = np.eye(4, dtype=int) * 5
    cm = _cm_from_contingency(contingency)
    expected = sk_emi(contingency, int(contingency.sum()))
    assert expected_mutual_info(cm) == pytest.approx(expected, rel=1e-10)


def test_string_labels():
    """Class labels need not be integers — the result depends only on row/col sums."""
    cm = metrics.ConfusionMatrix()
    pairs = [
        ("cat", "cat"),
        ("cat", "cat"),
        ("cat", "dog"),
        ("dog", "dog"),
        ("dog", "dog"),
        ("dog", "cat"),
        ("bird", "bird"),
        ("bird", "bird"),
        ("bird", "dog"),
    ]
    for yt, yp in pairs:
        cm.update(yt, yp)

    contingency = np.array([[2, 0, 1], [1, 0, 2], [0, 2, 1]])
    expected = sk_emi(contingency, int(contingency.sum()))
    assert expected_mutual_info(cm) == pytest.approx(expected, rel=1e-10)


def test_zero_row_or_col_is_skipped():
    """A class that's been seen but has zero sum (e.g. after revert) should not affect EMI."""
    cm = metrics.ConfusionMatrix()
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    for yt, yp in pairs:
        cm.update(yt, yp)
    cm.revert(2, 2)  # leaves class 2 with zero counts in both row and col
    assert cm.sum_row[2] == 0
    assert cm.sum_col[2] == 0

    contingency = np.array([[1, 1], [1, 1]])
    expected = sk_emi(contingency, int(contingency.sum()))
    assert expected_mutual_info(cm) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("seed", list(range(5)))
def test_random_matrices(seed):
    rng = random.Random(seed)
    n_classes = rng.randint(2, 6)
    n_clusters = rng.randint(2, 6)
    contingency = np.array(
        [[rng.randint(0, 8) for _ in range(n_clusters)] for _ in range(n_classes)],
    )
    # Skip degenerate matrices that would produce a zero-sum row or column.
    if (contingency.sum(axis=0) == 0).any() or (contingency.sum(axis=1) == 0).any():
        pytest.skip("degenerate contingency matrix")
    cm = _cm_from_contingency(contingency)
    expected = sk_emi(contingency, int(contingency.sum()))
    assert expected_mutual_info(cm) == pytest.approx(expected, rel=1e-10, abs=1e-12)


def test_large_counts():
    """Stress test with larger N to check numerical stability."""
    rng = random.Random(42)
    contingency = np.array(
        [[rng.randint(5, 40) for _ in range(4)] for _ in range(4)],
    )
    cm = _cm_from_contingency(contingency)
    expected = sk_emi(contingency, int(contingency.sum()))
    assert expected_mutual_info(cm) == pytest.approx(expected, rel=1e-10, abs=1e-12)
