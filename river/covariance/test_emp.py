from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import pytest

from river import covariance, stream


@pytest.mark.parametrize(
    "ddof",
    [
        pytest.param(
            ddof,
            id=f"{ddof=}",
        )
        for ddof in (0, 1, 2)
    ],
)
def test_covariance_revert(ddof):
    X = np.random.random((100, 5))
    X1 = X[: len(X) // 2]
    X2 = X[len(X) // 2 :]

    C1 = covariance.EmpiricalCovariance(ddof=ddof)
    for x, _ in stream.iter_array(X1):
        C1.update(x)

    C2 = covariance.EmpiricalCovariance(ddof=ddof)
    for x, _ in stream.iter_array(X):
        C2.update(x)
    for x, _ in stream.iter_array(X2):
        C2.revert(x)

    for k in C1._cov:
        assert math.isclose(C1._cov[k].get(), C2._cov[k].get())


@pytest.mark.parametrize(
    "ddof",
    [
        pytest.param(
            ddof,
            id=f"{ddof=}",
        )
        for ddof in (0, 1, 2)
    ],
)
def test_covariance_update_shuffled(ddof):
    C1 = covariance.EmpiricalCovariance(ddof=ddof)
    C2 = covariance.EmpiricalCovariance(ddof=ddof)

    X = np.random.random((100, 5))

    for x, _ in stream.iter_array(X):
        C1.update(x)
        C2.update({i: x[i] for i in random.sample(list(x.keys()), k=len(x))})

    for i, j in C1._cov:
        assert math.isclose(C1[i, j].get(), C2[i, j].get())


def test_covariance_update_sampled():
    # NOTE: this test only works with ddof=1 because pandas ignores it if there are missing values
    ddof = 1
    cov = covariance.EmpiricalCovariance(ddof=ddof)

    X = np.random.random((100, 5))
    samples = []

    for x, _ in stream.iter_array(X):
        sample = {i: x[i] for i in random.sample(list(x.keys()), k=len(x) - 1)}
        cov.update(sample)
        samples.append(sample)

    pd_cov = pd.DataFrame(samples).cov(ddof=ddof)

    for i, j in cov._cov:
        assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])


@pytest.mark.parametrize(
    "ddof",
    [
        pytest.param(
            ddof,
            id=f"{ddof=}",
        )
        for ddof in [0, 1]
    ],
)
def test_covariance_update_many(ddof):
    cov = covariance.EmpiricalCovariance(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(p):
        n = np.random.randint(1, 31)
        X = pd.DataFrame(np.random.random((n, p)))

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])


@pytest.mark.parametrize(
    "ddof",
    [
        pytest.param(
            ddof,
            id=f"{ddof=}",
        )
        for ddof in [0, 1]
    ],
)
def test_covariance_update_many_shuffled(ddof):
    cov = covariance.EmpiricalCovariance(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(p):
        n = np.random.randint(5, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(p, axis="columns")

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])


def test_covariance_update_many_sampled():
    # NOTE: this test only works with ddof=1 because pandas ignores it if there are missing values
    ddof = 1
    cov = covariance.EmpiricalCovariance(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(p):
        n = np.random.randint(5, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(p - 1, axis="columns")

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])


def test_precision_update_shuffled():
    C1 = covariance.EmpiricalPrecision()
    C2 = covariance.EmpiricalPrecision()

    X = np.random.random((100, 5))

    for x, _ in stream.iter_array(X):
        C1.update(x)
        C2.update({i: x[i] for i in random.sample(list(x.keys()), k=len(x))})

    for i, j in C1._inv_cov:
        assert math.isclose(C1[i, j], C2[i, j])


def test_precision_update_many_mini_batches():
    C1 = covariance.EmpiricalPrecision()
    C2 = covariance.EmpiricalPrecision()

    X = pd.DataFrame(np.random.random((100, 5)))

    C1.update_many(X)
    for Xb in np.split(X, 5):
        C2.update_many(Xb)

    for i, j in C1._inv_cov:
        assert math.isclose(C1[i, j], C2[i, j])


def test_precision_one_many_same():
    one = covariance.EmpiricalPrecision()
    many = covariance.EmpiricalPrecision()

    X = np.random.random((100, 5))

    for x, _ in stream.iter_array(X):
        one.update(x)
    many.update_many(pd.DataFrame(X))

    for i, j in one._inv_cov:
        assert math.isclose(one[i, j], many[i, j])
