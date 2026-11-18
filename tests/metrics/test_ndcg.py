from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from river import metrics, utils


def _random_queries(seed, n_queries=20):
    """Yield (relevances, ranking, scores) triples with distinct scores, so there are no ties."""
    rng = np.random.default_rng(seed)
    for _ in range(n_queries):
        n = int(rng.integers(2, 9))
        rel = rng.integers(0, 5, size=n)
        rel[rng.integers(n)] = rng.integers(1, 5)  # at least one relevant item
        ranking = rng.permutation(n)  # ranking[0] is the item ranked first
        scores = np.zeros(n)
        scores[ranking] = np.arange(n, 0, -1)
        yield rel, [int(i) for i in ranking], scores


def _update(metric, rel, ranking):
    metric.update(dict(enumerate(int(r) for r in rel)), ranking)


@pytest.mark.parametrize("k", [None, 1, 3, 5, 20])
@pytest.mark.parametrize("seed", range(5))
def test_ndcg_against_sklearn(k, seed):
    metric, expected = metrics.NDCG(k=k), []
    for rel, ranking, scores in _random_queries(seed):
        expected.append(sk_metrics.ndcg_score([rel], [scores], k=k))
        _update(metric, rel, ranking)
    assert metric.get() == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("k", [None, 1, 3, 5, 20])
@pytest.mark.parametrize("seed", range(5))
def test_dcg_against_sklearn(k, seed):
    metric, expected = metrics.DCG(k=k), []
    for rel, ranking, scores in _random_queries(seed):
        expected.append(sk_metrics.dcg_score([rel], [scores], k=k))
        _update(metric, rel, ranking)
    assert metric.get() == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("k", [None, 1, 3, 5, 20])
@pytest.mark.parametrize("seed", range(5))
def test_cg_against_reference(k, seed):
    # scikit-learn has no cumulative gain function, so the reference is a plain sum of the
    # relevances of the first k ranked items
    metric, expected = metrics.CumulativeGain(k=k), []
    for rel, ranking, _ in _random_queries(seed):
        expected.append(rel[ranking[:k]].sum())
        _update(metric, rel, ranking)
    assert metric.get() == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("seed", range(3))
def test_rolling_against_sklearn(seed):
    window = 4
    queries = list(_random_queries(seed, n_queries=10))
    rolling = utils.Rolling(metrics.NDCG, window_size=window, k=3)
    for i, (rel, ranking, _) in enumerate(queries):
        rolling.update(dict(enumerate(int(r) for r in rel)), ranking)
        recent = queries[max(0, i - window + 1) : i + 1]
        expected = np.mean([sk_metrics.ndcg_score([r], [s], k=3) for r, _, s in recent])
        assert rolling.get() == pytest.approx(expected)


def test_perfect_and_worst():
    metric = metrics.NDCG()
    metric.update({"a": 2, "b": 1}, ["a", "b"])
    assert metric.get() == pytest.approx(1.0)

    metric = metrics.NDCG()
    metric.update({"a": 2, "b": 1}, ["x", "y"])
    assert metric.get() == 0.0


def test_no_relevant_item_scores_zero():
    metric = metrics.NDCG()
    metric.update({"a": 0}, ["a"])
    assert metric.get() == 0.0


def test_collection_of_relevant_items():
    assert metrics.NDCG()._eval({"a", "b"}, ["a", "b"]) == pytest.approx(1.0)
    assert metrics.NDCG()._eval(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_invalid_k():
    with pytest.raises(ValueError):
        metrics.NDCG(k=0)


def test_revert():
    metric = metrics.NDCG()
    metric.update({"a": 1}, ["a"])
    before = metric.get()
    metric.update({"a": 1}, ["b", "a"])
    metric.revert({"a": 1}, ["b", "a"])
    assert metric.get() == pytest.approx(before)


def test_rolling():
    rolling = utils.Rolling(metrics.NDCG, window_size=1)
    rolling.update({"a": 1}, ["a"])
    rolling.update({"a": 1}, ["b"])
    assert rolling.get() == 0.0


def test_dcg_by_hand():
    metric = metrics.DCG()
    metric.update({"a": 3, "b": 2, "c": 1}, ["a", "b", "c"])
    assert metric.get() == pytest.approx(3 / 1 + 2 / 1.5849625 + 1 / 2)


def test_dcg_first_position_is_not_discounted():
    metric = metrics.DCG()
    metric.update({"a": 5}, ["a"])
    assert metric.get() == 5.0


def test_dcg_k():
    metric = metrics.DCG(k=1)
    metric.update({"a": 3, "b": 2}, ["b", "a"])
    assert metric.get() == 2.0


def test_ndcg_is_dcg_over_ideal():
    y_true, y_pred = {"a": 3, "b": 2, "c": 1}, ["c", "b", "a"]
    dcg, ndcg, ideal = metrics.DCG(), metrics.NDCG(), metrics.DCG()
    dcg.update(y_true, y_pred)
    ndcg.update(y_true, y_pred)
    ideal.update(y_true, ["a", "b", "c"])
    assert ndcg.get() == pytest.approx(dcg.get() / ideal.get())


def test_cg_by_hand():
    metric = metrics.CumulativeGain()
    metric.update({"a": 3, "b": 2, "c": 1}, ["c", "b", "a"])
    assert metric.get() == 6.0


def test_cg_ignores_order():
    a, b = metrics.CumulativeGain(), metrics.CumulativeGain()
    a.update({"a": 3, "b": 2}, ["a", "b"])
    b.update({"a": 3, "b": 2}, ["b", "a"])
    assert a.get() == b.get() == 5.0


def test_cg_k_and_missing_items():
    metric = metrics.CumulativeGain(k=2)
    metric.update({"a": 3, "b": 2, "c": 1}, ["x", "b", "a"])
    assert metric.get() == 2.0


def test_cg_revert_and_invalid_k():
    metric = metrics.CumulativeGain()
    metric.update({"a": 1}, ["a"])
    metric.update({"a": 1}, ["b", "a"])
    metric.revert({"a": 1}, ["b", "a"])
    assert metric.get() == 1.0
    with pytest.raises(ValueError):
        metrics.CumulativeGain(k=0)
