from __future__ import annotations

import math
import random

import scipy.stats

from river import stats, utils


def test_chi_squared_basic() -> None:
    chi2 = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi2.update(x, y)

    assert chi2.get() > 0
    assert chi2.degrees_of_freedom == 1
    assert math.isclose(chi2.p_value, 0.04550026389635842)


def test_chi_squared_zero_case() -> None:
    chi2 = stats.ChiSquared()

    # No dependency: all belong to the same y value
    data = [
        ("A", 0),
        ("B", 0),
        ("C", 0),
    ]

    for x, y in data:
        chi2.update(x, y)

    assert chi2.get() == 0.0
    assert chi2.degrees_of_freedom == 0
    assert chi2.p_value == 1.0


def test_chi_squared_exact_value() -> None:
    chi2 = stats.ChiSquared()

    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi2.update(x, y)

    assert math.isclose(chi2.get(), 4.0)


def test_chi_squared_multiple_categories() -> None:
    chi2 = stats.ChiSquared()

    data = [
        ("A", 0),
        ("B", 1),
        ("C", 0),
        ("A", 1),
        ("B", 0),
        ("C", 1),
    ]

    for x, y in data:
        chi2.update(x, y)

    assert chi2.get() >= 0
    assert chi2.degrees_of_freedom == 2


def test_chi_squared_empty() -> None:
    chi2 = stats.ChiSquared()
    assert chi2.get() == 0.0
    assert chi2.degrees_of_freedom == 0
    assert chi2.p_value == 1.0


def test_chi_squared_revert() -> None:
    chi2 = stats.ChiSquared()
    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
    ]

    for x, y in data:
        chi2.update(x, y)

    # Revert one observation
    chi2.revert("B", 1)

    # Expected contingency table:
    # A: {0: 2}
    # B: {1: 1}
    # n = 3
    # y_totals: {0: 2, 1: 1}
    # x_totals: {A: 2, B: 1}
    #
    # E(A, 0) = 2 * 2 / 3 = 1.333
    # E(A, 1) = 2 * 1 / 3 = 0.666
    # E(B, 0) = 1 * 2 / 3 = 0.666
    # E(B, 1) = 1 * 1 / 3 = 0.333
    # O(A, 0)=2, O(A, 1)=0, O(B, 0)=0, O(B, 1)=1
    # chi2 = (2 - 1.333)^2 / 1.333 + (0 - 0.666)^2 / 0.666
    #      + (0 - 0.666)^2 / 0.666 + (1 - 0.333)^2 / 0.333
    #      = 0.333 + 0.666 + 0.666 + 1.333 = 3.0

    assert math.isclose(chi2.get(), 3.0)

    # Revert everything
    chi2.revert("A", 0)
    chi2.revert("A", 0)
    chi2.revert("B", 1)

    assert chi2.get() == 0.0
    assert chi2.n == 0
    assert len(chi2.counts) == 0


def test_chi_squared_rolling() -> None:
    chi2 = utils.Rolling(stats.ChiSquared, window_size=4)
    data = [
        ("A", 0),
        ("A", 0),
        ("B", 1),
        ("B", 1),
        ("C", 0),  # This will push out ("A", 0)
    ]

    for x, y in data[:4]:
        chi2.update(x, y)
    # `utils.Rolling` proxies attribute access via `__getattr__ -> object`, so mypy cannot
    # know that `get` is callable here.
    assert math.isclose(chi2.get(), 4.0)  # type: ignore[operator]

    chi2.update(*data[4])
    # Now window is [("A", 0), ("B", 1), ("B", 1), ("C", 0)]
    # A: {0: 1}, B: {1: 2}, C: {0: 1}
    # n=4
    # y_totals: {0: 2, 1: 2}
    # x_totals: {A: 1, B: 2, C: 1}
    # E(A, 0) = 1 * 2 / 4 = 0.5, E(A, 1) = 0.5
    # E(B, 0) = 2 * 2 / 4 = 1.0, E(B, 1) = 1.0
    # E(C, 0) = 0.5, E(C, 1) = 0.5
    # chi2 = (1 - 0.5)^2 / 0.5 + (0 - 0.5)^2 / 0.5
    #      + (0 - 1.0)^2 / 1.0 + (2 - 1.0)^2 / 1.0
    #      + (1 - 0.5)^2 / 0.5 + (0 - 0.5)^2 / 0.5
    #      = 0.5 + 0.5 + 1.0 + 1.0 + 0.5 + 0.5 = 4.0
    assert math.isclose(chi2.get(), 4.0)  # type: ignore[operator]


def test_chi_squared_scipy_comparison() -> None:
    chi2 = stats.ChiSquared()

    random.seed(42)
    n = 100
    x_values = [random.choice(["A", "B", "C"]) for _ in range(n)]
    y_values = [random.choice([0, 1]) for _ in range(n)]

    for xi, yi in zip(x_values, y_values):
        chi2.update(xi, yi)

    # Get contingency table for scipy
    categories = sorted(set(x_values))
    classes = sorted(set(y_values))
    contingency = [
        [sum(1 for xv, yv in zip(x_values, y_values) if xv == cat and yv == cl) for cl in classes]
        for cat in categories
    ]

    result = scipy.stats.chi2_contingency(contingency, correction=False)
    scipy_chi2, scipy_p, scipy_df = float(result[0]), float(result[1]), int(result[2])

    assert math.isclose(chi2.get(), scipy_chi2)
    assert math.isclose(chi2.p_value, scipy_p)
    assert chi2.degrees_of_freedom == scipy_df
