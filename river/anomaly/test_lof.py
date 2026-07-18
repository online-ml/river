from __future__ import annotations

import numpy as np
import pytest
from sklearn import neighbors

from river import anomaly
from river.conftest import FRAME_BACKENDS, FrameBackend

np.random.seed(42)


def _blobs():
    """Three dense Gaussian blobs plus scattered outliers, as feature dicts."""
    norm = 0.5 * np.random.rand(60, 2)
    inliers = np.concatenate((norm - 2, norm, norm + 2), axis=0)
    outliers = np.random.uniform(low=-4, high=4, size=(15, 2))
    X = np.concatenate((inliers, outliers), axis=0)
    return X, [{"a": row[0], "b": row[1]} for row in X]


def test_score_matches_sklearn_static():
    """Scoring stored points equals scikit-learn's in-sample LOF (`negative_outlier_factor_`).

    A learned point is excluded from its own neighborhood, so learn-then-score reproduces the
    classic batch LOF computed over the same samples.
    """
    X, dicts = _blobs()
    k = 20

    lof = anomaly.LocalOutlierFactor(n_neighbors=k)
    for x in dicts:
        lof.learn_one(x)
    river_scores = np.array([lof.score_one(x) for x in dicts])

    sklearn_lof = neighbors.LocalOutlierFactor(n_neighbors=k)
    sklearn_lof.fit_predict(X)
    sklearn_scores = -sklearn_lof.negative_outlier_factor_

    assert np.allclose(river_scores, sklearn_scores, rtol=1e-7, atol=1e-7)


def test_score_matches_sklearn_novelty():
    """Scoring unseen points equals scikit-learn's novelty LOF over the learned window."""
    X, dicts = _blobs()
    k = 20

    queries = np.random.RandomState(0).uniform(-3, 3, size=(25, 2))
    query_dicts = [{"a": row[0], "b": row[1]} for row in queries]

    lof = anomaly.LocalOutlierFactor(n_neighbors=k)
    for x in dicts:
        lof.learn_one(x)
    river_scores = np.array([lof.score_one(q) for q in query_dicts])

    sklearn_lof = neighbors.LocalOutlierFactor(n_neighbors=k, novelty=True).fit(X)
    sklearn_scores = -sklearn_lof.score_samples(queries)

    assert np.allclose(river_scores, sklearn_scores, rtol=1e-7, atol=1e-7)


def test_empty_model_scores_zero():
    lof = anomaly.LocalOutlierFactor()
    assert lof.score_one({"a": 1.0, "b": 2.0}) == 0.0


def test_score_one_does_not_learn():
    """Scoring must not modify the window (no implicit learning)."""
    _, dicts = _blobs()
    lof = anomaly.LocalOutlierFactor(n_neighbors=5)
    for x in dicts:
        lof.learn_one(x)

    before = list(lof._nn.window)
    lof.score_one({"a": 0.0, "b": 0.0})
    assert list(lof._nn.window) == before


def test_duplicates_are_handled():
    """Identical points must not break learning or scoring (former issues #1328 / #1331)."""
    lof = anomaly.LocalOutlierFactor()
    for _ in range(5):
        lof.learn_one({"a": 1.0, "b": 1.0})
    # A point coinciding with a dense duplicate cluster is a strong inlier (score <= 1).
    assert lof.score_one({"a": 1.0, "b": 1.0}) <= 1.0


def test_learn_many_matches_learn_one(frame_backend: FrameBackend):
    """Row-by-row `learn_one` and batched `learn_many` yield the same scores on every backend."""
    _, dicts = _blobs()
    cols = {c: [x[c] for x in dicts] for c in dicts[0]}

    one = anomaly.LocalOutlierFactor(n_neighbors=10)
    for x in dicts:
        one.learn_one(x)

    many = anomaly.LocalOutlierFactor(n_neighbors=10)
    many.learn_many(frame_backend.frame(cols))

    for x in dicts:
        assert one.score_one(x) == pytest.approx(many.score_one(x))


def test_learn_many_is_backend_agnostic(frame_backend: FrameBackend):
    """`learn_many` produces identical scores regardless of the dataframe backend."""
    _, dicts = _blobs()
    cols = {c: [x[c] for x in dicts] for c in dicts[0]}

    reference = anomaly.LocalOutlierFactor(n_neighbors=10)
    reference.learn_many(FRAME_BACKENDS["pandas"]().frame(cols))

    model = anomaly.LocalOutlierFactor(n_neighbors=10)
    model.learn_many(frame_backend.frame(cols))

    for x in dicts:
        assert model.score_one(x) == pytest.approx(reference.score_one(x))


def test_window_bounds_memory():
    """An engine with a small window keeps only the most recent samples."""
    from river import neighbors

    lof = anomaly.LocalOutlierFactor(n_neighbors=5, engine=neighbors.LazySearch(window_size=30))
    _, dicts = _blobs()
    for x in dicts:
        lof.learn_one(x)
    assert len(lof._nn.window) == 30
