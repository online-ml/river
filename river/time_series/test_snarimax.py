from __future__ import annotations

import calendar
import math
import random

import pytest
import sympy

from river import compose, datasets, metrics, time_series
from river.time_series.snarimax import Differencer


class Yt(sympy.IndexedBase):
    t = sympy.symbols("t", cls=sympy.Idx)

    def __getitem__(self, idx):
        return super().__getitem__(self.t - idx)


def test_diff_formula():
    """

    >>> import sympy
    >>> from river.time_series.snarimax import Differencer

    >>> Y = Yt('y')
    >>> Y
    y

    >>> p = sympy.symbols('p')
    >>> p
    p

    >>> D = Differencer

    1
    >>> D(0).diff(p, Y)
    p

    (1 - B)
    >>> D(1).diff(p, Y)
    p - y[t]

    (1 - B)^2
    >>> D(2).diff(p, Y)
    p + y[t - 1] - 2*y[t]

    (1 - B^m)
    >>> m = sympy.symbols('m', cls=sympy.Idx)
    >>> D(1, m).diff(p, Y)
    p - y[-m + t + 1]

    (1 - B)(1 - B^m)
    >>> (D(1) * D(1, m)).diff(p, Y)
    p - y[-m + t + 1] + y[-m + t] - y[t]

    (1 - B)(1 - B^12)
    >>> (D(1) * D(1, 12)).diff(p, Y)
    p - y[t - 11] + y[t - 12] - y[t]

    """


def test_diff_example():
    """https://people.duke.edu/~rnau/411sdif.htm

    >>> import pandas as pd
    >>> from river.time_series.snarimax import Differencer

    >>> sales = pd.DataFrame([
    ...     {'date': 'Jan-70', 'autosale': 4.79, 'cpi': 0.297},
    ...     {'date': 'Feb-70', 'autosale': 4.96, 'cpi': 0.298},
    ...     {'date': 'Mar-70', 'autosale': 5.64, 'cpi': 0.300},
    ...     {'date': 'Apr-70', 'autosale': 5.98, 'cpi': 0.302},
    ...     {'date': 'May-70', 'autosale': 6.08, 'cpi': 0.303},
    ...     {'date': 'Jun-70', 'autosale': 6.55, 'cpi': 0.305},
    ...     {'date': 'Jul-70', 'autosale': 6.11, 'cpi': 0.306},
    ...     {'date': 'Aug-70', 'autosale': 5.37, 'cpi': 0.306},
    ...     {'date': 'Sep-70', 'autosale': 5.17, 'cpi': 0.308},
    ...     {'date': 'Oct-70', 'autosale': 5.48, 'cpi': 0.309},
    ...     {'date': 'Nov-70', 'autosale': 4.49, 'cpi': 0.311},
    ...     {'date': 'Dec-70', 'autosale': 4.65, 'cpi': 0.312},
    ...     {'date': 'Jan-71', 'autosale': 5.17, 'cpi': 0.312},
    ...     {'date': 'Feb-71', 'autosale': 5.57, 'cpi': 0.313},
    ...     {'date': 'Mar-71', 'autosale': 6.92, 'cpi': 0.314},
    ...     {'date': 'Apr-71', 'autosale': 7.10, 'cpi': 0.315},
    ...     {'date': 'May-71', 'autosale': 7.02, 'cpi': 0.316},
    ...     {'date': 'Jun-71', 'autosale': 7.58, 'cpi': 0.319},
    ...     {'date': 'Jul-71', 'autosale': 6.93, 'cpi': 0.319},
    ... ])

    >>> sales['autosale/cpi'] = sales.eval('autosale / cpi').round(2)
    >>> Y = sales['autosale/cpi'].to_list()

    >>> diff = Differencer(1)
    >>> sales['(1 - B)'] = [
    ...     diff.diff(p, Y[:i][::-1])
    ...     if i else ''
    ...     for i, p in enumerate(Y)
    ... ]

    >>> sdiff = Differencer(1, 12)
    >>> sales['(1 - B^12)'] = [
    ...     sdiff.diff(p, Y[:i][::-1])
    ...     if i >= 12 else ''
    ...     for i, p in enumerate(Y)
    ... ]

    >>> sales['(1 - B)(1 - B^12)'] = [
    ...     (diff * sdiff).diff(p, Y[:i][::-1])
    ...     if i >= 13 else ''
    ...     for i, p in enumerate(Y)
    ... ]

    >>> sales
          date  autosale    cpi  autosale/cpi (1 - B) (1 - B^12) (1 - B)(1 - B^12)
    0   Jan-70      4.79  0.297         16.13
    1   Feb-70      4.96  0.298         16.64    0.51
    2   Mar-70      5.64  0.300         18.80    2.16
    3   Apr-70      5.98  0.302         19.80     1.0
    4   May-70      6.08  0.303         20.07    0.27
    5   Jun-70      6.55  0.305         21.48    1.41
    6   Jul-70      6.11  0.306         19.97   -1.51
    7   Aug-70      5.37  0.306         17.55   -2.42
    8   Sep-70      5.17  0.308         16.79   -0.76
    9   Oct-70      5.48  0.309         17.73    0.94
    10  Nov-70      4.49  0.311         14.44   -3.29
    11  Dec-70      4.65  0.312         14.90    0.46
    12  Jan-71      5.17  0.312         16.57    1.67       0.44
    13  Feb-71      5.57  0.313         17.80    1.23       1.16              0.72
    14  Mar-71      6.92  0.314         22.04    4.24       3.24              2.08
    15  Apr-71      7.10  0.315         22.54     0.5       2.74              -0.5
    16  May-71      7.02  0.316         22.22   -0.32       2.15             -0.59
    17  Jun-71      7.58  0.319         23.76    1.54       2.28              0.13
    18  Jul-71      6.93  0.319         21.72   -2.04       1.75             -0.53

    """


@pytest.mark.parametrize(
    "differencer",
    [
        Differencer(1),
        Differencer(2),
        Differencer(1, 2),
        Differencer(2, 2),
        Differencer(1, 10),
        Differencer(2, 10),
        Differencer(1) * Differencer(1),
        Differencer(2) * Differencer(1),
        Differencer(1) * Differencer(2),
        Differencer(1) * Differencer(1, 2),
        Differencer(2) * Differencer(1, 2),
        Differencer(1, 2) * Differencer(1, 10),
        Differencer(1, 2) * Differencer(2, 10),
        Differencer(2, 2) * Differencer(1, 10),
        Differencer(2, 2) * Differencer(2, 10),
    ],
)
def test_undiff(differencer):
    Y = [random.random() for _ in range(max(differencer.coeffs))]
    p = random.random()

    diffed = differencer.diff(p, Y)
    undiffed = differencer.undiff(diffed, Y)
    assert math.isclose(undiffed, p)


@pytest.mark.parametrize(
    "snarimax, Y, errors, expected",
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
def test_add_lag_features(snarimax, Y, errors, expected):
    features = snarimax._add_lag_features(x=None, Y=Y, errors=errors)
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
            calendar.month_name[month]: math.exp(-((x["month"].month - month) ** 2))
            for month in range(1, 13)
        }

    def get_ordinal_date(x):
        return {"ordinal_date": x["month"].toordinal()}

    extract_features = compose.TransformerUnion(get_ordinal_date, get_month_distances)

    model = extract_features | snarimax

    time_series.evaluate(
        dataset=datasets.AirlinePassengers(), model=model, metric=metrics.MAE(), horizon=12
    )
