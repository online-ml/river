import functools
import math
import random
import statistics

import numpy as np
import pytest

from river import stats


@pytest.mark.parametrize(
    "stat1, stat2, func",
    [
        (stats.Mean(), stats.Mean(), statistics.mean),
        (stats.Var(ddof=0), stats.Var(ddof=0), np.var),
        (stats.Var(), stats.Var(), functools.partial(np.var, ddof=1)),
    ],
)
def test_add_mean_var(stat1, stat2, func):
    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]

    for i, (x, y) in enumerate(zip(X, Y)):
        stat1.update(x)
        stat2.update(y)
        if i >= 1:
            assert math.isclose(
                (stat1 + stat2).get(), func(X[: i + 1] + Y[: i + 1]), abs_tol=1e-10
            )

    stat1 += stat2
    assert math.isclose(stat1.get(), func(X + Y), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat1, stat2, func",
    [
        (stats.Mean(), stats.Mean(), statistics.mean),
        (stats.Var(ddof=0), stats.Var(ddof=0), np.var),
        (stats.Var(), stats.Var(), functools.partial(np.var, ddof=1)),
    ],
)
def test_sub_mean_var(stat1, stat2, func):
    X = [random.random() for _ in range(30)]

    for x in X:
        stat1.update(x)

    for i, x in enumerate(X):
        stat2.update(x)
        if i < len(X) - 2:
            assert math.isclose((stat1 - stat2).get(), func(X[i + 1 :]), abs_tol=1e-10)

    # Test inplace subtraction
    X.extend(random.random() for _ in range(3))
    for i in range(30, 33):
        stat1.update(X[i])

    stat1 -= stat2
    assert math.isclose(stat1.get(), func(X[30:33]), abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat1, stat2, func",
    [
        (stats.Cov(ddof=0), stats.Cov(ddof=0), functools.partial(np.cov, ddof=0)),
        (stats.Cov(ddof=1), stats.Cov(ddof=1), functools.partial(np.cov, ddof=1)),
    ],
)
def test_add_cov(stat1, stat2, func):
    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]

    W = [random.random() for _ in range(30)]
    Z = [random.random() for _ in range(30)]

    for i, (x, y, w, z) in enumerate(zip(X, Y, W, Z)):
        stat1.update(x, y)
        stat2.update(w, z)
        if i >= 1:
            assert math.isclose(
                (stat1 + stat2).get(),
                func(X[: i + 1] + W[: i + 1], Y[: i + 1] + Z[: i + 1])[0, 1],
                abs_tol=1e-10,
            )

    stat1 += stat2
    assert math.isclose(stat1.get(), func(X + W, Y + Z)[0, 1], abs_tol=1e-10)


@pytest.mark.parametrize(
    "stat1, stat2, func",
    [
        (stats.Cov(ddof=0), stats.Cov(ddof=0), functools.partial(np.cov, ddof=0)),
        (stats.Cov(ddof=1), stats.Cov(ddof=1), functools.partial(np.cov, ddof=1)),
    ],
)
def test_sub_cov(stat1, stat2, func):
    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]

    for x, y in zip(X, Y):
        stat1.update(x, y)

    for i, (x, y) in enumerate(zip(X, Y)):
        stat2.update(x, y)
        if i < len(X) - 2:
            assert math.isclose(
                (stat1 - stat2).get(), func(X[i + 1 :], Y[i + 1 :])[0, 1], abs_tol=1e-10
            )

    # Test inplace subtraction
    X.extend(random.random() for _ in range(3))
    Y.extend(random.random() for _ in range(3))
    for i in range(30, 33):
        stat1.update(X[i], Y[i])

    stat1 -= stat2
    assert math.isclose(stat1.get(), func(X[30:33], Y[30:33])[0, 1], abs_tol=1e-10)
