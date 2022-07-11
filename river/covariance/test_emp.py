import math

import numpy as np
import pandas as pd

from river import misc


def test_covariance_revert():

    X1 = X[:len(X) // 2]
    X2 = X[len(X) // 2:]

    C1 = covariance.EmpiricalCovariance()
    for x, _ in stream.iter_array(X1):
        C1.update(x)

    C2 = covariance.EmpiricalCovariance()
    for x, _ in stream.iter_array(X):
        C2.update(x)
    for x, _ in stream.iter_array(X2):
        C2.revert(x)

    np.testing.assert_array_almost_equal(C1, C2)


def test_cov_matrix():

    # NOTE: this test only works with ddof=1 because pandas ignores it if there are missing values
    ddof = 1

    cov = misc.CovMatrix(ddof=ddof)
    p = 5
    X_all = pd.DataFrame(columns=range(p))

    for _ in range(5):
        n = np.random.randint(1, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(3, axis="columns")
        cov.update_many(X)
        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._covs:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])
