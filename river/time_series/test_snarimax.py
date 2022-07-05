import pytest

from river import time_series


@pytest.mark.parametrize(
    "snarimax, y_trues, errors, expected",
    [
        # Non-seasonal parts (p and q)
        (
            time_series.SNARIMAX(p=3, d=0, q=3),
            [1, 2, 3],
            [-4, -5, -6],
            {"e-1": -4, "e-2": -5, "e-3": -6, "y-1": 1, "y-2": 2, "y-3": 3},
        ),
        (
            time_series.SNARIMAX(p=2, d=0, q=3),
            [1, 2, 3],
            [-4, -5, -6],
            {"e-1": -4, "e-2": -5, "e-3": -6, "y-1": 1, "y-2": 2},
        ),
        (
            time_series.SNARIMAX(p=3, d=0, q=2),
            [1, 2, 3],
            [-4, -5, -6],
            {"e-1": -4, "e-2": -5, "y-1": 1, "y-2": 2, "y-3": 3},
        ),
        (
            time_series.SNARIMAX(p=2, d=0, q=2),
            [1, 2, 3],
            [-4, -5, -6],
            {"e-1": -4, "e-2": -5, "y-1": 1, "y-2": 2},
        ),
        # Not enough data, so features too far away are omitted
        (
            time_series.SNARIMAX(p=3, d=0, q=3),
            [1, 2],
            [-4, -5],
            {"e-1": -4, "e-2": -5, "y-1": 1, "y-2": 2},
        ),
        # Seasonal AR
        (
            time_series.SNARIMAX(p=2, d=0, q=2, m=3, sp=2),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "e-2": 1, "sy-3": 2, "sy-6": 5, "y-1": 0, "y-2": 1},
        ),
        (
            time_series.SNARIMAX(p=2, d=0, q=2, m=2, sp=2),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "e-2": 1, "sy-2": 1, "sy-4": 3, "y-1": 0, "y-2": 1},
        ),
        (
            time_series.SNARIMAX(p=2, d=0, q=2, m=2, sp=3),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "e-2": 1, "sy-2": 1, "sy-4": 3, "sy-6": 5, "y-1": 0, "y-2": 1},
        ),
        # Seasonal MA
        (
            time_series.SNARIMAX(p=2, d=0, q=2, m=3, sq=2),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "e-2": 1, "se-3": 2, "se-6": 5, "y-1": 0, "y-2": 1},
        ),
        (
            time_series.SNARIMAX(p=2, d=0, q=2, m=3, sq=4),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "e-2": 1, "se-3": 2, "se-6": 5, "se-9": 8, "se-12": 11, "y-1": 0, "y-2": 1},
        ),
        (
            time_series.SNARIMAX(p=1, d=0, q=1, m=2, sq=4),
            [i for i in range(12)],
            [i for i in range(12)],
            {"e-1": 0, "se-2": 1, "se-4": 3, "se-6": 5, "se-8": 7, "y-1": 0},
        ),
    ],
)
def test_add_lag_features(snarimax, y_trues, errors, expected):
    features = snarimax._add_lag_features(x=None, y_trues=y_trues, errors=errors)
    assert features == expected
