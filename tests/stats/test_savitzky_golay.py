from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.signal import savgol_coeffs

from river import stats


def test_savitzky_golay_matches_least_squares_polyfit() -> None:
    window_size, polyorder = 7, 3
    stat = stats.SavitzkyGolay(window_size=window_size, polyorder=polyorder)
    x = [float(i) for i in range(20)]

    coeffs = savgol_coeffs(
        window_length=window_size,
        polyorder=polyorder,
        pos=window_size - 1,
        use="dot",
    )

    for i, value in enumerate(x):
        stat.update(value)
        if i < window_size - 1:
            assert stat.get() is None
            continue
        window = x[i - window_size + 1 : i + 1]
        expected = sum(c * v for c, v in zip(coeffs, window))
        assert math.isclose(stat.get(), expected, rel_tol=1e-10)
        assert stat.get() == np.dot(coeffs, window)


def test_savitzky_golay_preserves_polynomials() -> None:
    # A polynomial of degree <= polyorder should pass through unchanged once the
    # window is full, up to floating point error.
    window_size, polyorder = 5, 2
    stat = stats.SavitzkyGolay(window_size=window_size, polyorder=polyorder)

    x = [float(i) ** 2 for i in range(10)]
    for i, value in enumerate(x):
        stat.update(value)
        if i >= window_size - 1:
            assert math.isclose(stat.get(), value, rel_tol=1e-8)


def test_savitzky_golay_returns_none_until_window_is_full() -> None:
    stat = stats.SavitzkyGolay(window_size=3, polyorder=1)
    for i in range(2):
        stat.update(float(i))
        assert stat.get() is None
    stat.update(2.0)
    assert stat.get() is not None


def test_savitzky_golay_requires_polyorder_smaller_than_window() -> None:
    with pytest.raises(ValueError):
        stats.SavitzkyGolay(window_size=3, polyorder=3)
