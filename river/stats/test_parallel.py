import copy
import functools
import math
import random

import numpy as np
import pytest

from river import stats


@pytest.mark.parametrize(
    "stat",
    [
        pytest.param(stat, id=stat.__class__.__name__)
        for stat in [stats.Mean(), stats.Var(ddof=0), stats.Var(ddof=1)]
    ],
)
def test_add(stat):
    A = copy.deepcopy(stat)
    B = copy.deepcopy(stat)
    C = copy.deepcopy(stat)

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]
    W = [random.random() for _ in range(30)]

    for x, y, w in zip(X, Y, W):
        A.update(x, w)
        B.update(y, w)
        C.update(x, w).update(y, w)

    D = A + B
    assert math.isclose(C.get(), D.get())

    A += B
    assert math.isclose(C.get(), A.get())


@pytest.mark.parametrize(
    "stat",
    [
        pytest.param(stat, id=stat.__class__.__name__)
        for stat in [stats.Mean(), stats.Var(ddof=0), stats.Var(ddof=1)]
    ],
)
def test_sub(stat):
    A = copy.deepcopy(stat)
    B = copy.deepcopy(stat)
    C = copy.deepcopy(stat)

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]
    W = [random.random() for _ in range(30)]

    for x, y, w in zip(X, Y, W):
        A.update(x, w)
        B.update(y, w)
        C.update(x, w).update(y, w)

    D = C - B
    assert math.isclose(D.get(), A.get())

    C -= B
    assert math.isclose(C.get(), A.get())


@pytest.mark.parametrize(
    "stat",
    [
        pytest.param(stat, id=stat.__class__.__name__)
        for stat in [stats.Mean(), stats.Var(ddof=0), stats.Var(ddof=1)]
    ],
)
def test_sub_back_to_zero(stat):

    A = copy.deepcopy(stat)
    B = copy.deepcopy(stat)
    C = copy.deepcopy(stat)

    x = random.random()
    A.update(x)
    B.update(x)

    D = A - B
    assert math.isclose(D.get(), C.get())


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
