import math

import numpy as np
import pandas as pd

from river import misc


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

        for i, j in cov:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])
