from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd
import pytest

from river import covariance, stream


def _pd_split(df, n):
    """Split a pandas DataFrame or Series into n chunks without triggering swapaxes deprecation."""
    indices = np.array_split(range(len(df)), n)
    return [df.iloc[idx] for idx in indices]


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
    X_all = None

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
    X_all = None

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
    X_all = None

    for _ in range(p):
        n = np.random.randint(5, 31)
        X = pd.DataFrame(np.random.random((n, p))).sample(p - 1, axis="columns")

        cov.update_many(X)

        X_all = pd.concat((X_all, X)).astype(float)
        pd_cov = X_all.cov(ddof=ddof)

        for i, j in cov._cov:
            assert math.isclose(cov[i, j].get(), pd_cov.loc[i, j])


def test_covariance_update_many_backend_agnostic(frame_backend):
    """`update_many` yields identical covariances across every dataframe backend."""
    np.random.seed(0)
    data = {col: np.random.random(40).tolist() for col in ["a", "b", "c"]}

    reference = covariance.EmpiricalCovariance()
    reference.update_many(pd.DataFrame(data))

    cov = covariance.EmpiricalCovariance()
    cov.update_many(frame_backend.frame(data))

    assert cov.matrix.keys() == reference.matrix.keys()
    for key in reference.matrix:
        assert math.isclose(cov[key].get(), reference[key].get(), rel_tol=1e-9, abs_tol=1e-12)


def test_precision_update_many_backend_agnostic(frame_backend):
    """`update_many` yields identical precisions across every dataframe backend."""
    np.random.seed(0)
    data = {col: np.random.random(60).tolist() for col in ["a", "b", "c"]}

    reference = covariance.EmpiricalPrecision()
    reference.update_many(pd.DataFrame(data))

    prec = covariance.EmpiricalPrecision()
    prec.update_many(frame_backend.frame(data))

    for i, j in reference.matrix:
        assert math.isclose(prec[i, j], reference[i, j], rel_tol=1e-9, abs_tol=1e-12)


def test_precision_update_shuffled():
    C1 = covariance.EmpiricalPrecision()
    C2 = covariance.EmpiricalPrecision()

    X = np.random.random((100, 5))

    for x, _ in stream.iter_array(X):
        C1.update(x)
        C2.update({i: x[i] for i in random.sample(list(x.keys()), k=len(x))})

    for i, j in C1.matrix:
        assert math.isclose(C1[i, j], C2[i, j])


def test_precision_update_many_mini_batches():
    C1 = covariance.EmpiricalPrecision()
    C2 = covariance.EmpiricalPrecision()

    X = pd.DataFrame(np.random.random((100, 5)))

    C1.update_many(X)
    for Xb in _pd_split(X, 5):
        C2.update_many(Xb)

    for i, j in C1.matrix:
        assert math.isclose(C1[i, j], C2[i, j])


def test_precision_one_many_same():
    one = covariance.EmpiricalPrecision()
    many = covariance.EmpiricalPrecision()

    X = np.random.random((100, 5))

    for x, _ in stream.iter_array(X):
        one.update(x)
    many.update_many(pd.DataFrame(X))

    for i, j in one.matrix:
        assert math.isclose(one[i, j], many[i, j])


def test_covariance_emerging_features():
    """New features should be registered the first time they appear."""
    np.random.seed(0)
    cov = covariance.EmpiricalCovariance()

    # Phase 1: only features 0, 1
    for _ in range(50):
        cov.update({0: np.random.random(), 1: np.random.random()})

    # Phase 2: feature 2 appears
    for _ in range(50):
        cov.update({i: np.random.random() for i in range(3)})

    # Diagonal counts reflect each feature's appearance history (not co-appearance)
    assert cov[0, 0].n == 100
    assert cov[1, 1].n == 100
    assert cov[2, 2].n == 50
    # Pairwise count reflects co-appearance
    assert cov[0, 1].n == 100
    assert cov[0, 2].n == 50
    assert cov[1, 2].n == 50


def test_covariance_radically_shifting_features():
    """`EmpiricalCovariance` must not crash when consecutive samples share no features."""
    cov = covariance.EmpiricalCovariance()
    cov.update({"a": 1.0, "b": 2.0})
    cov.update({"c": 3.0, "d": 4.0})
    cov.update({"a": 5.0, "d": 6.0})

    assert cov["a", "a"].n == 2
    assert cov["b", "b"].n == 1
    assert cov["c", "c"].n == 1
    assert cov["d", "d"].n == 2
    # (a, b) seen once; (c, d) seen once; (a, d) seen once
    assert cov["a", "b"].n == 1
    assert cov["a", "d"].n == 1
    # (a, c) was never co-observed
    with pytest.raises(KeyError):
        cov["a", "c"]


def test_covariance_empty():
    """A fresh `EmpiricalCovariance` has an empty matrix and rejects lookups."""
    cov = covariance.EmpiricalCovariance()
    assert dict(cov.matrix) == {}
    with pytest.raises(KeyError):
        cov["a", "b"]


def test_covariance_single_sample():
    """A single sample yields zero variance / covariance entries."""
    cov = covariance.EmpiricalCovariance()
    cov.update({"a": 1.0, "b": 2.0})
    assert cov["a", "a"].n == 1
    assert cov["a", "a"].get() == 0.0
    assert cov["a", "b"].n == 1
    assert cov["a", "b"].get() == 0.0


def test_covariance_from_state_roundtrip():
    """`_from_state` reconstructs a covariance object whose values match a batch fit."""
    ddof = 1
    np.random.seed(0)
    X = pd.DataFrame(np.random.random((50, 3)), columns=list("abc"))

    reference = covariance.EmpiricalCovariance(ddof=ddof)
    reference.update_many(X)

    n = len(X)
    mean = {col: X[col].mean() for col in X.columns}
    pd_cov = X.cov(ddof=ddof)
    cov_dict = {(i, j): pd_cov.loc[i, j] for i in X.columns for j in X.columns}

    rebuilt = covariance.EmpiricalCovariance._from_state(n=n, mean=mean, cov=cov_dict, ddof=ddof)

    for key in reference.matrix:
        assert math.isclose(reference[key].get(), rebuilt[key].get(), rel_tol=1e-9)


def test_precision_emerging_features():
    """`EmpiricalPrecision` should grow its internal matrix when features appear."""
    np.random.seed(0)
    prec = covariance.EmpiricalPrecision()

    for _ in range(50):
        prec.update({0: np.random.random(), 1: np.random.random()})
    assert len(prec._idx) == 2

    for _ in range(50):
        prec.update({i: np.random.random() for i in range(3)})
    assert len(prec._idx) == 3

    # All matrix entries finite and symmetric (within tolerance)
    for i in range(3):
        for j in range(3):
            assert math.isfinite(prec[i, j])
            assert math.isclose(prec[i, j], prec[j, i], rel_tol=1e-9, abs_tol=1e-12)


def test_precision_update_sampled():
    """`EmpiricalPrecision` should accept samples where one feature is missing each round."""
    random.seed(0)
    np.random.seed(0)
    prec = covariance.EmpiricalPrecision()

    X = np.random.random((100, 5))
    for row in X:
        x = {i: row[i] for i in range(5)}
        x.pop(random.choice(list(x)))
        prec.update(x)

    # All co-observed feature pairs return finite values
    for i in range(5):
        for j in range(5):
            assert math.isfinite(prec[i, j])


def test_precision_empty():
    """A fresh `EmpiricalPrecision` has an empty matrix and rejects lookups."""
    prec = covariance.EmpiricalPrecision()
    assert prec.matrix == {}
    with pytest.raises(KeyError):
        prec["a", "b"]


def test_precision_single_sample():
    """A single sample should leave the precision matrix finite."""
    prec = covariance.EmpiricalPrecision()
    prec.update({"a": 1.0, "b": 2.0})
    for i, j in [("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")]:
        assert math.isfinite(prec[i, j])


def test_precision_symmetric_access():
    """`prec[a, b]` and `prec[b, a]` should agree to machine precision."""
    np.random.seed(0)
    prec = covariance.EmpiricalPrecision()
    for _ in range(200):
        prec.update({i: np.random.random() for i in range(4)})
    for i in range(4):
        for j in range(i + 1, 4):
            assert math.isclose(prec[i, j], prec[j, i], rel_tol=1e-9, abs_tol=1e-12)


def test_repr_runs():
    """`__repr__` should produce a non-empty string for both classes."""
    cov = covariance.EmpiricalCovariance()
    prec = covariance.EmpiricalPrecision()
    for _ in range(10):
        x = {0: np.random.random(), 1: np.random.random()}
        cov.update(x)
        prec.update(x)
    assert isinstance(repr(cov), str) and len(repr(cov)) > 0
    assert isinstance(repr(prec), str) and len(repr(prec)) > 0
