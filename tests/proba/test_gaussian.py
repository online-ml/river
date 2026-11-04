from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd
import pytest

from river import proba


def assert_gaussians_equal(left, right):
    assert left.n_samples == pytest.approx(right.n_samples)
    assert left.mu == pytest.approx(right.mu)
    assert left.sigma == pytest.approx(right.sigma)


def test_scalar_updates_match_numpy():
    values = np.array([-4.0, -1.0, 0.5, 2.0, 8.0])
    gaussian = proba.Gaussian()

    for value in values:
        gaussian.update(value)

    assert gaussian.n_samples == len(values)
    assert gaussian.mu == pytest.approx(np.mean(values))
    assert gaussian.sigma == pytest.approx(np.std(values, ddof=1))


def test_weighted_update_and_revert():
    weighted = proba.Gaussian()
    repeated = proba.Gaussian()

    for value, weight in [(1.0, 2), (3.0, 1), (-2.0, 4)]:
        weighted.update(value, weight)
        for _ in range(weight):
            repeated.update(value)

    assert_gaussians_equal(weighted, repeated)

    weighted.revert(-2.0, 3)
    for _ in range(3):
        repeated.revert(-2.0)

    assert_gaussians_equal(weighted, repeated)


def test_batch_updates_match_scalar_updates():
    values = np.array([1.0, -3.0, 5.0, 7.0, 0.5])
    batch = proba.Gaussian()
    scalar = proba.Gaussian()

    batch.update(11.0)
    batch._update_many(values)
    scalar.update(11.0)
    for value in values:
        scalar.update(value)

    assert_gaussians_equal(batch, scalar)
    assert batch._pdf_many(values) == pytest.approx([batch(value) for value in values])


def test_batch_pdf_of_degenerate_distribution_is_zero():
    gaussian = proba.Gaussian()
    gaussian.update(1.0)

    assert np.array_equal(gaussian._pdf_many(np.array([0.0, 1.0, 2.0])), np.zeros(3))


@pytest.mark.parametrize("ddof", [0, 1])
def test_from_state(ddof):
    gaussian = proba.Gaussian._from_state(n=5, m=3, var=4, ddof=ddof)

    assert gaussian.n_samples == 5
    assert gaussian.mu == 3
    assert gaussian.sigma == 2
    assert gaussian(3) == pytest.approx(1 / math.sqrt(8 * math.pi))
    assert gaussian.log_pdf(3) == pytest.approx(math.log(gaussian(3)))
    assert gaussian.cdf(3) == pytest.approx(0.5)


def test_pickle_round_trip():
    gaussian = proba.Gaussian(seed=42)
    gaussian._update_many(np.array([1.0, 2.0, 4.0, 8.0]))
    restored = pickle.loads(pickle.dumps(gaussian))

    assert_gaussians_equal(restored, gaussian)
    assert restored(3.0) == pytest.approx(gaussian(3.0))
    assert restored.cdf(3.0) == pytest.approx(gaussian.cdf(3.0))
    assert restored.sample() == gaussian.sample()


def test_zero_weight_is_a_noop():
    gaussian = proba.Gaussian()
    gaussian.update(1.0)
    gaussian.update(100.0, 0.0)
    gaussian.revert(100.0, 0.0)

    assert gaussian.n_samples == 1
    assert gaussian.mu == 1
    assert gaussian.sigma == 0


def test_revert_cannot_make_count_negative():
    gaussian = proba.Gaussian()
    gaussian.update(1.0)

    with pytest.raises(ValueError, match="Count cannot go below 0"):
        gaussian.revert(1.0, 2.0)


@pytest.mark.parametrize(
    ("value", "weight"),
    [(math.nan, 1.0), (math.inf, 1.0), (1.0, -1.0), (1.0, math.inf)],
)
def test_invalid_updates_are_rejected(value, weight):
    gaussian = proba.Gaussian()

    with pytest.raises(ValueError):
        gaussian.update(value, weight)


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
