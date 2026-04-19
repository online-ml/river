from __future__ import annotations

import numpy as np
import scipy.stats

from river import stats


def test_chi_squared_basic():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() > 0
    assert chi.degrees_of_freedom == 1
    assert np.isclose(chi.p_value, 0.04550026389635842)
    assert chi.is_significant(alpha=0.05)


def test_chi_squared_zero_case():
    chi = stats.ChiSquared()

    # No dependency: all belong to same class
    data = [
        ("A", 0),
        ("B", 0),
        ("C", 0),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() == 0.0
    assert chi.degrees_of_freedom == 0
    assert chi.p_value == 1.0


def test_chi_squared_incremental():
    chi = stats.ChiSquared()

    chi.update("A", 0)
    first = chi.get()

    chi.update("A", 1)
    second = chi.get()

    assert first >= 0
    assert second >= 0


def test_chi_squared_exact_value():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    # Known correct value
    assert abs(chi.get() - 4.0) < 1e-6


def test_chi_squared_multiple_categories():
    chi = stats.ChiSquared()

    data = [
        ("A", 0),
        ("B", 1),
        ("C", 0),
        ("A", 1),
        ("B", 0),
        ("C", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    assert chi.get() >= 0
    assert chi.degrees_of_freedom == 2


def test_chi_squared_empty():
    chi = stats.ChiSquared()
    assert chi.get() == 0.0
    assert chi.degrees_of_freedom == 0
    assert chi.p_value == 1.0


def test_chi_squared_revert():
    chi = stats.ChiSquared()
    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi.update(x, y)

    # Revert one observation
    chi.revert("B", 1)

    # Expected contingency table:
    # A: {0: 2}
    # B: {1: 1}
    # n = 3
    # class_totals: {0: 2, 1: 1}
    # value_totals: {A: 2, B: 1}

    # Chi-squared for this:
    # E(A,0) = 2*2 / 3 = 1.333
    # E(A,1) = 2*1 / 3 = 0.666
    # E(B,0) = 1*2 / 3 = 0.666
    # E(B,1) = 1*1 / 3 = 0.333
    # O(A,0)=2, O(A,1)=0, O(B,0)=0, O(B,1)=1
    # Chi2 = (2-1.333)^2/1.333 + (0-0.666)^2/0.666 + (0-0.666)^2/0.666 + (1-0.333)^2/0.333
    # Chi2 = 0.333 + 0.666 + 0.666 + 1.333 = 3.0

    assert np.isclose(chi.get(), 3.0)

    # Revert everything
    chi.revert("A", 0)
    chi.revert("A", 0)
    chi.revert("B", 1)

    assert chi.get() == 0.0
    assert chi.n == 0
    assert len(chi.counts) == 0


def test_chi_squared_rolling():
    chi = stats.RollingChiSquared(window_size=4)
    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
        ("C", 0),  # This will push out ("A", 0)
    ]

    for x, y in data[:4]:
        chi.update(x, y)
    assert np.isclose(chi.get(), 4.0)

    chi.update(*data[4])
    # Now window is [("A", 0), ("B", 1), ("B", 1), ("C", 0)]
    # A: {0: 1}, B: {1: 2}, C: {0: 1}
    # n=4
    # class_totals: {0: 2, 1: 2}
    # value_totals: {A: 1, B: 2, C: 1}
    # E(A,0) = 1*2/4 = 0.5, E(A,1) = 0.5
    # E(B,0) = 2*2/4 = 1.0, E(B,1) = 1.0
    # E(C,0) = 0.5, E(C,1) = 0.5
    # Chi2 = (1-0.5)^2/0.5 + (0-0.5)^2/0.5 + (0-1.0)^2/1.0 + (2-1.0)^2/1.0 + (1-0.5)^2/0.5 + (0-0.5)^2/0.5
    # Chi2 = 0.5 + 0.5 + 1.0 + 1.0 + 0.5 + 0.5 = 4.0
    assert np.isclose(chi.get(), 4.0)


def test_chi_squared_scipy_comparison():
    chi = stats.ChiSquared()

    # Generate some random categorical data
    np.random.seed(42)
    n = 100
    x = np.random.choice(["A", "B", "C"], size=n)
    y = np.random.choice([0, 1], size=n)

    for xi, yi in zip(x, y):
        chi.update(xi, yi)

    # Get contingency table for scipy
    categories = sorted(list(set(x)))
    classes = sorted(list(set(y)))
    contingency = []
    for cat in categories:
        row = []
        for cl in classes:
            row.append(np.sum((x == cat) & (y == cl)))
        contingency.append(row)

    scipy_chi2, scipy_p, scipy_df, _ = scipy.stats.chi2_contingency(contingency, correction=False)

    assert np.isclose(chi.get(), scipy_chi2)
    assert np.isclose(chi.p_value, scipy_p)
    assert chi.degrees_of_freedom == scipy_df
