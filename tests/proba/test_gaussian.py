from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from river import proba


@pytest.mark.parametrize(
    "p",
    [
        pytest.param(
            p,
            id=f"{p=}",
        )
        for p in [1, 3, 5]
    ],
)
def test_univariate_multivariate_consistency(p):
    X = pd.DataFrame(np.random.random((30, p)), columns=range(p))

    multi = proba.MultivariateGaussian()
    single = {c: proba.Gaussian() for c in X.columns}

    for x in X.to_dict(orient="records"):
        multi.update(x)
        for c, s in single.items():
            s.update(x[c])

    for c in X.columns:
        assert math.isclose(multi.mu[c], single[c].mu)
        assert math.isclose(multi.sigma[c][c], single[c].sigma)


def test_multivariate_cdf_is_order_independent():
    p = proba.MultivariateGaussian()

    data = [
        {"a": 1.0, "b": 5.0, "c": 10.0},
        {"a": 2.0, "b": 10.0, "c": 11.0},
        {"a": 3.0, "b": 15.0, "c": 9.0},
        {"a": 4.0, "b": 20.0, "c": 12.0},
        {"a": 5.0, "b": 25.0, "c": 10.0},
    ]

    for x in data:
        p.update(x)

    x = {"a": 2.5, "b": 12.5, "c": 10.5}
    x_reordered = {"c": 10.5, "a": 2.5, "b": 12.5}

    assert p.cdf(x) == pytest.approx(p.cdf(x_reordered), rel=1e-2)
