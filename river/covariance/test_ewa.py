from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from river import covariance, stats, stream


def _dense(cov):
    """Materialize a SymmetricMatrix into a dense numpy array (public-API only)."""
    names = sorted({i for i, _ in cov.matrix})

    def value(entry):
        # EmpiricalCovariance stores stats.Cov/Var objects; EWA stores plain floats.
        return entry.get() if isinstance(entry, stats.base.Statistic) else entry

    return np.array([[value(cov[i, j]) for j in names] for i in names], dtype=float)


def _ewa_reference(X, fading_factor):
    """Independent exponentially weighted covariance, matching the stats.EWMean convention."""
    f = fading_factor
    mean = M2 = None
    for v in X:
        v = np.asarray(v, dtype=float)
        if mean is None:
            mean, M2 = v.copy(), np.outer(v, v)
        else:
            mean = (1 - f) * mean + f * v
            M2 = (1 - f) * M2 + f * np.outer(v, v)
    return M2 - np.outer(mean, mean)


def _precision_reference(X, fading_factor):
    """Independent EW precision: inverse of the EW second-moment matrix (identity prior)."""
    f = fading_factor
    d = X.shape[1]
    M2 = np.eye(d)
    mean = None
    for v in X:
        v = np.asarray(v, dtype=float)
        M2 = (1 - f) * M2 + f * np.outer(v, v)
        mean = v.copy() if mean is None else (1 - f) * mean + f * v
    return np.linalg.inv(M2 - np.outer(mean, mean))


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    true_cov = [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
    return rng.multivariate_normal(mean=[0, 0, 0], cov=true_cov, size=300)


# --------------------------------------------------------------------- EwaCovariance core


def test_ewa_matches_numpy_reference(returns):
    cov = covariance.EwaCovariance(fading_factor=0.05)
    for x, _ in stream.iter_array(returns):
        cov.update(x)
    np.testing.assert_allclose(_dense(cov), _ewa_reference(returns, 0.05))


def test_ewa_diagonal_matches_ewvar_and_offdiag_ewcov(returns):
    f = 0.05
    cov = covariance.EwaCovariance(fading_factor=f)
    ewvars = [stats.EWVar(fading_factor=f) for _ in range(3)]
    ewcov_01 = stats.EWCov(fading_factor=f)
    for x, _ in stream.iter_array(returns):
        cov.update(x)
        for i in range(3):
            ewvars[i].update(x[i])
        ewcov_01.update(x[0], x[1])

    for i in range(3):
        assert cov[i, i] == pytest.approx(ewvars[i].get())
    assert cov[0, 1] == pytest.approx(ewcov_01.get())


@pytest.mark.parametrize(
    "estimator",
    [
        covariance.EwaCovariance,
        covariance.EwaPrecision,
        covariance.LedoitWolfCovariance,
        covariance.OASCovariance,
        covariance.ShrunkCovariance,
    ],
)
def test_single_equals_minibatch(returns, estimator):
    one_by_one = estimator()
    for x, _ in stream.iter_array(returns):
        one_by_one.update(x)

    batched = estimator()
    batched.update_many(pd.DataFrame(returns, columns=[0, 1, 2]))

    np.testing.assert_allclose(_dense(one_by_one), _dense(batched))


def test_minibatch_backend_agnostic(returns):
    pl = pytest.importorskip("polars")
    df = pd.DataFrame(returns, columns=["a", "b", "c"])

    pandas_cov = covariance.EwaCovariance(fading_factor=0.05)
    pandas_cov.update_many(df)

    polars_cov = covariance.EwaCovariance(fading_factor=0.05)
    polars_cov.update_many(pl.from_pandas(df))

    np.testing.assert_allclose(_dense(pandas_cov), _dense(polars_cov))


@pytest.mark.parametrize(
    "estimator",
    [
        covariance.EmpiricalCovariance,
        covariance.EmpiricalPrecision,
        covariance.EwaCovariance,
        covariance.EwaPrecision,
        covariance.ShrunkCovariance,
    ],
)
def test_empty_repr_does_not_crash(estimator):
    assert repr(estimator()) == f"{estimator.__name__} (empty)"


def test_missing_feature_raises():
    cov = covariance.EwaCovariance()
    cov.update({"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="missing feature"):
        cov.update({"a": 1.0})


# --------------------------------------------------------------------- EwaPrecision


def test_ewa_precision_matches_reference(returns):
    prec = covariance.EwaPrecision(fading_factor=0.05)
    for x, _ in stream.iter_array(returns):
        prec.update(x)
    np.testing.assert_allclose(_dense(prec), _precision_reference(returns, 0.05))


def test_ewa_precision_inverts_covariance():
    # Enough samples that the decaying identity prior (~ (1 - f)^n) is numerically negligible.
    rng = np.random.default_rng(2)
    X = rng.multivariate_normal(
        mean=[0, 0, 0], cov=[[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]], size=2000
    )
    f = 0.05
    cov = covariance.EwaCovariance(fading_factor=f)
    prec = covariance.EwaPrecision(fading_factor=f)
    for x, _ in stream.iter_array(X):
        cov.update(x)
        prec.update(x)
    S, P = _dense(cov), _dense(prec)
    np.testing.assert_allclose(P @ S, np.eye(S.shape[0]), atol=1e-9)


def test_ewa_precision_requires_strict_fading_factor():
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            covariance.EwaPrecision(fading_factor=bad)


# --------------------------------------------------------------------- shrinkage estimators


@pytest.mark.parametrize(
    "estimator",
    [
        covariance.LedoitWolfCovariance,
        covariance.OASCovariance,
        covariance.EwaPrecision,
        lambda: covariance.ShrunkCovariance(target="identity", delta=0.2),
        lambda: covariance.ShrunkCovariance(target="constant_correlation", delta=0.2),
    ],
)
def test_matrix_is_symmetric(returns, estimator):
    cov = estimator()
    for x, _ in stream.iter_array(returns):
        cov.update(x)
    M = _dense(cov)
    np.testing.assert_allclose(M, M.T)


@pytest.mark.parametrize(
    "estimator",
    [
        covariance.LedoitWolfCovariance,
        covariance.OASCovariance,
        lambda: covariance.ShrunkCovariance(target="identity", delta=0.2),
    ],
)
def test_shrinkage_towards_identity_is_psd(returns, estimator):
    cov = estimator()
    for x, _ in stream.iter_array(returns):
        cov.update(x)
    eigvals = np.linalg.eigvalsh(_dense(cov))
    assert np.all(eigvals > -1e-9)


def test_shrunk_identity_matches_sklearn(returns):
    from sklearn.covariance import shrunk_covariance

    f, delta = 0.05, 0.3
    raw = covariance.EwaCovariance(fading_factor=f)
    shrunk = covariance.ShrunkCovariance(fading_factor=f, delta=delta, target="identity")
    for x, _ in stream.iter_array(returns):
        raw.update(x)
        shrunk.update(x)

    expected = shrunk_covariance(_dense(raw), shrinkage=delta)
    np.testing.assert_allclose(_dense(shrunk), expected)


def test_shrunk_delta_bounds(returns):
    f = 0.05
    raw = covariance.EwaCovariance(fading_factor=f)
    no_shrink = covariance.ShrunkCovariance(fading_factor=f, delta=0.0, target="identity")
    full_shrink = covariance.ShrunkCovariance(fading_factor=f, delta=1.0, target="identity")
    for x, _ in stream.iter_array(returns):
        raw.update(x)
        no_shrink.update(x)
        full_shrink.update(x)

    S = _dense(raw)
    np.testing.assert_allclose(_dense(no_shrink), S)
    mu = np.trace(S) / S.shape[0]
    np.testing.assert_allclose(_dense(full_shrink), mu * np.eye(S.shape[0]), atol=1e-9)


def test_ledoitwolf_is_convex_combination_towards_identity(returns):
    f = 0.05
    raw = covariance.EwaCovariance(fading_factor=f)
    lw = covariance.LedoitWolfCovariance(fading_factor=f)
    for x, _ in stream.iter_array(returns):
        raw.update(x)
        lw.update(x)

    S, L = _dense(raw), _dense(lw)
    mu = np.trace(S) / S.shape[0]
    # Recover the intensity from an off-diagonal entry (the identity target is 0 there).
    delta = 1 - L[0, 1] / S[0, 1]
    assert 0.0 <= delta <= 1.0
    np.testing.assert_allclose(L, (1 - delta) * S + delta * mu * np.eye(S.shape[0]), atol=1e-9)


def test_oas_matches_formula(returns):
    f = 0.05
    raw = covariance.EwaCovariance(fading_factor=f)
    oas = covariance.OASCovariance(fading_factor=f)
    for x, _ in stream.iter_array(returns):
        raw.update(x)
        oas.update(x)

    S = _dense(raw)
    d = S.shape[0]
    n = max(1.0 / f, 2.0)
    tr, tr2 = np.trace(S), np.trace(S @ S)
    mu = tr / d
    num = (1 - 2 / d) * tr2 + tr * tr
    den = (n + 1 - 2 / d) * (tr2 - tr * tr / d)
    rho = 1.0 if den <= 0 else min(max(num / den, 0.0), 1.0)
    np.testing.assert_allclose(_dense(oas), (1 - rho) * S + rho * mu * np.eye(d))


def test_shrinkage_improves_conditioning():
    # Many variables, short effective memory: the raw covariance is ill-conditioned.
    rng = np.random.default_rng(1)
    true_cov = 0.7 * np.ones((12, 12)) + 0.3 * np.eye(12)
    X = rng.multivariate_normal(mean=np.zeros(12), cov=true_cov, size=200)

    raw = covariance.EwaCovariance(fading_factor=0.05)
    lw = covariance.LedoitWolfCovariance(fading_factor=0.05)
    oas = covariance.OASCovariance(fading_factor=0.05)
    for x, _ in stream.iter_array(X):
        raw.update(x)
        lw.update(x)
        oas.update(x)

    raw_cond = np.linalg.cond(_dense(raw))
    assert np.linalg.cond(_dense(lw)) < raw_cond
    assert np.linalg.cond(_dense(oas)) < raw_cond


@pytest.mark.parametrize(
    "estimator",
    [
        covariance.EwaCovariance,
        covariance.EwaPrecision,
        covariance.LedoitWolfCovariance,
        covariance.OASCovariance,
        covariance.ShrunkCovariance,
    ],
)
def test_pickle_round_trip(returns, estimator):
    import pickle

    cov = estimator()
    for x, _ in stream.iter_array(returns):
        cov.update(x)
    restored = pickle.loads(pickle.dumps(cov))
    np.testing.assert_allclose(_dense(restored), _dense(cov))


# --------------------------------------------------------------------- EmpiricalCovariance


def test_empirical_covariance_matches_sklearn(returns):
    # sklearn's EmpiricalCovariance is the maximum-likelihood estimate, i.e. ddof=0.
    sklearn_cov = pytest.importorskip("sklearn.covariance")

    cov = covariance.EmpiricalCovariance(ddof=0)
    for x, _ in stream.iter_array(returns):
        cov.update(x)

    expected = sklearn_cov.EmpiricalCovariance().fit(returns).covariance_
    np.testing.assert_allclose(_dense(cov), expected)


# --------------------------------------------------------------------- stats.EWCov


def test_ewcov_matches_pandas():
    # pandas computes the exponentially weighted covariance with its own routine rather than
    # via the E[xy] - E[x]E[y] identity used by stats.EWCov, so this is an independent check.
    # adjust=False matches the recursive stats.EWMean convention; bias=True matches the
    # population (uncorrected) covariance that the identity yields.
    f = 0.3
    x = [1.0, 3.0, 5.0, 4.0, 6.0]
    y = [2.0, 4.0, 3.0, 6.0, 5.0]
    ewcov = stats.EWCov(fading_factor=f)

    values = []
    for xi, yi in zip(x, y):
        ewcov.update(xi, yi)
        values.append(ewcov.get())

    df = pd.DataFrame({"x": x, "y": y})
    expected = df["x"].ewm(alpha=f, adjust=False).cov(df["y"], bias=True)
    np.testing.assert_allclose(values, expected.to_numpy())
