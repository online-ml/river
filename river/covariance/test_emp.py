import math

import numpy as np
import pandas as pd

import pytest
from river import covariance, stream
from sklearn import datasets


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
    X1 = X[:len(X) // 2]
    X2 = X[len(X) // 2:]

    C1 = covariance.EmpiricalCovariance(ddof=ddof)
    for x, _ in stream.iter_array(X1):
        C1.update(x)

    C2 = covariance.EmpiricalCovariance(ddof=ddof)
    for x, _ in stream.iter_array(X):
        C2.update(x)
    for x, _ in stream.iter_array(X2):
        C2.revert(x)

    for k in C1._cov:
        assert math.isclose(C1._cov[k], C2._cov[k])


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
def test_covariance_update_many_shuffled_columns(ddof):

    cov = covariance.EmpiricalCovariance(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(5):
        n = np.random.randint(1, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(p, axis="columns")

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j], pd_cov.loc[i, j])


def test_covariance_update_many_sampled_columns():

    # NOTE: this test only works with ddof=1 because pandas ignores it if there are missing values
    ddof = 1
    cov = covariance.EmpiricalCovariance(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(5):
        n = np.random.randint(1, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(p - 1, axis="columns")

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j], pd_cov.loc[i, j])
