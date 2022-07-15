import calendar
import math
import pytest

from river import compose,datasets,metrics,time_series

import sympy
class Yt(sympy.IndexedBase):
    t = sympy.symbols('t', cls=sympy.Idx)

    def __getitem__(self, idx):
        return super().__getitem__(self.t - idx)


def test_differencing():
    """

    >>> import sympy
    >>> from river.time_series.snarimax import Differencer

    >>> import sympy

    >>> Y = Yt('y')
    >>> m = sympy.symbols('m', cls=sympy.Idx)

    >>> D = Differencer

    1
    >>> D(0).diff(Y)
    y[t]

    (1 - B)
    >>> D(1).diff(Y)
    -y[t - 1] + y[t]

    (1 - B)^2
    >>> D(2).diff(Y)
    -2*y[t - 1] + y[t - 2] + y[t]

    (1 - B^m)
    >>> D(1, m).diff(Y)
    -y[-m + t] + y[t]

    (1 - B)(1 - B^m)
    >>> (D(1) * D(1, m)).diff(Y)
    y[-m + t - 1] - y[-m + t] - y[t - 1] + y[t]

    (1 - B)(1 - B^12)
    >>> (D(1) * D(1, 12)).diff(Y)
    -y[t - 12] + y[t - 13] - y[t - 1] + y[t]

    """


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


@pytest.mark.parametrize(
    "snarimax",
    [
        time_series.SNARIMAX(p=1, d=1, q=0, m=12, sp=0, sd=1, sq=0),
        time_series.SNARIMAX(p=0, d=1, q=0, m=12, sp=1, sd=1, sq=0),
        time_series.SNARIMAX(p=1, d=2, q=0, m=12, sp=0, sd=0, sq=0),
        time_series.SNARIMAX(p=1, d=0, q=0, m=12, sp=0, sd=2, sq=0),
    ],
)
def test_no_overflow(snarimax):

    def get_month_distances(x):
        return {
            calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2)
            for month in range(1, 13)
        }

    def get_ordinal_date(x):
        return {'ordinal_date': x['month'].toordinal()}

    extract_features = compose.TransformerUnion(
        get_ordinal_date,
        get_month_distances
    )

    model = (
        extract_features |
        snarimax
    )

    time_series.evaluate(
        dataset=datasets.AirlinePassengers(),
        model=model,
        metric=metrics.MAE(),
        horizon=12
    )
