import itertools
import math

import numpy as np
import pandas as pd
import pytest

from river import misc


@pytest.mark.parametrize("ddof", [0, 1, 2], ids=lambda ddof: f"ddof={ddof}")
def test_cov_matrix(ddof):

    cov = misc.CovMatrix(ddof)
    k = 5
    X_all = pd.DataFrame(columns=range(k))

    for _ in range(5):
        X = pd.DataFrame(np.random.random((30, k))).sample(3, axis="columns")
        cov.update_many(X)
        X_all = pd.concat((X_all, X))

        for i, j in itertools.combinations_with_replacement(range(k), r=2):
            not_null = X_all[i].notnull() & X_all[j].notnull()
            if not not_null.any():
                continue
            np_cov = np.cov(X_all.loc[not_null, [i, j]], rowvar=False, ddof=ddof)
            assert math.isclose(cov[i, j].get(), np_cov[0, 1])
