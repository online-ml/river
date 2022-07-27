import copy
import math
import random

import numpy as np
import pytest

from river import stats


def test_add_mean():
    A = stats.Mean()
    B = stats.Mean()
    C = stats.Mean()

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]
    W = [random.random() for _ in range(30)]

    for i, (x, y, w) in enumerate(zip(X, Y, W)):
        A.update(x, w)
        B.update(y, w)
        C.update(x, w).update(y, w)

        D = A + B
        assert math.isclose(C.get(), D.get())

        # TODO: clone should work instead of deepcopy
        E = copy.deepcopy(A)
        E += B
        assert math.isclose(C.get(), E.get())

        assert math.isclose(
            C.get(), np.average(X[: i + 1] + Y[: i + 1], weights=W[: i + 1] + W[: i + 1])
        )


def _weighted_var(X, W, ddof):
    average = np.average(X, weights=W)
    return np.average((X - average) ** 2, weights=W) * len(X) / (len(X) - ddof)


@pytest.mark.parametrize("ddof", [pytest.param(ddof, id=f"{ddof=}") for ddof in [0, 1, 2]])
def test_add_var(ddof):
    A = stats.Var(ddof)
    B = stats.Var(ddof)
    C = stats.Var(ddof)

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]
    W = [1 for _ in range(30)]

    for i, (x, y, w) in enumerate(zip(X, Y, W)):
        A.update(x, w)
        B.update(y, w)
        C.update(x, w).update(y, w)

        D = A + B
        assert math.isclose(C.get(), D.get())

        E = A.clone(include_attributes=True)
        E += B
        assert math.isclose(C.get(), E.get())

        if i >= ddof:
            assert math.isclose(
                C.get(), _weighted_var(X[: i + 1] + Y[: i + 1], W[: i + 1] + W[: i + 1], ddof=ddof)
            )


@pytest.mark.parametrize(
    "stat",
    [
        pytest.param(stat, id=stat.__class__.__name__)
        for stat in [stats.Mean(), stats.Var(ddof=0), stats.Var(ddof=1), stats.Var(ddof=2)]
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
        for stat in [stats.Mean(), stats.Var(ddof=0), stats.Var(ddof=1), stats.Var(ddof=2)]
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
    "ddof",
    [pytest.param(ddof, id=f"{ddof=}") for ddof in [0, 1]],
)
def test_add_cov(ddof):

    c1 = stats.Cov(ddof=ddof)
    c2 = stats.Cov(ddof=ddof)

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]

    W = [random.random() for _ in range(30)]
    Z = [random.random() for _ in range(30)]

    for i, (x, y, w, z) in enumerate(zip(X, Y, W, Z)):
        c1.update(x, y)
        c2.update(w, z)
        if i >= c1.ddof:
            assert math.isclose(
                c1.get(),
                np.cov(X[: i + 1], Y[: i + 1], ddof=ddof)[0, 1],
                abs_tol=1e-10,
            )
            assert math.isclose(
                c2.get(),
                np.cov(W[: i + 1], Z[: i + 1], ddof=ddof)[0, 1],
                abs_tol=1e-10,
            )
            assert math.isclose(
                (c1 + c2).get(),
                np.cov(X[: i + 1] + W[: i + 1], Y[: i + 1] + Z[: i + 1], ddof=ddof)[0, 1],
                abs_tol=1e-10,
            )

    c1 += c2
    assert math.isclose(c1.get(), np.cov(X + W, Y + Z, ddof=ddof)[0, 1], abs_tol=1e-10)


@pytest.mark.parametrize(
    "ddof",
    [pytest.param(ddof, id=f"{ddof=}") for ddof in [0, 1]],
)
def test_sub_cov(ddof):

    c1 = stats.Cov(ddof=ddof)
    c2 = stats.Cov(ddof=ddof)

    X = [random.random() for _ in range(30)]
    Y = [random.random() for _ in range(30)]

    for x, y in zip(X, Y):
        c1.update(x, y)

    for i, (x, y) in enumerate(zip(X, Y)):
        c2.update(x, y)
        if i < len(X) - 2:
            assert math.isclose(
                (c1 - c2).get(), np.cov(X[i + 1 :], Y[i + 1 :], ddof=ddof)[0, 1], abs_tol=1e-10
            )

    # Test inplace subtraction
    X.extend(random.random() for _ in range(3))
    Y.extend(random.random() for _ in range(3))
    for i in range(30, 33):
        c1.update(X[i], Y[i])

    c1 -= c2
    assert math.isclose(c1.get(), np.cov(X[30:33], Y[30:33], ddof=ddof)[0, 1], abs_tol=1e-10)
