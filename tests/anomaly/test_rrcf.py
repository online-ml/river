from __future__ import annotations

import collections
import pickle
import random

from river import anomaly


def _synthetic_stream(n_normal=400, n_anom=20, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_normal):
        rows.append(({"a": rng.gauss(0.0, 1.0), "b": rng.gauss(0.0, 1.0)}, 0))
    for _ in range(n_anom):
        rows.append(({"a": rng.gauss(12.0, 1.0), "b": rng.gauss(-12.0, 1.0)}, 1))
    rng.shuffle(rows)
    return rows


def test_learn_and_score_shape():
    rrcf = anomaly.RobustRandomCutForest(n_trees=10, tree_size=64, seed=42)
    rrcf.learn_one({"a": 1.0, "b": 2.0})
    score = rrcf.score_one({"a": 1.0, "b": 2.0})
    assert isinstance(score, float)
    assert score >= 0.0


def test_score_before_any_learn_is_zero():
    rrcf = anomaly.RobustRandomCutForest(seed=42)
    assert rrcf.score_one({"a": 1.0, "b": 2.0}) == 0.0


def test_roc_auc_floor():
    from sklearn.metrics import roc_auc_score

    rrcf = anomaly.RobustRandomCutForest(n_trees=20, tree_size=128, seed=42)
    scores, labels = [], []
    for x, y in _synthetic_stream():
        scores.append(rrcf.score_one(x))
        rrcf.learn_one(x)
        labels.append(y)
    assert roc_auc_score(labels, scores) > 0.8


def test_score_one_is_pure():
    rrcf = anomaly.RobustRandomCutForest(n_trees=10, tree_size=64, seed=1)
    for x, _ in _synthetic_stream(n_normal=120, n_anom=5, seed=3):
        rrcf.learn_one(x)

    x = {"a": 3.0, "b": -3.0}
    before = pickle.dumps(rrcf)
    for _ in range(5):
        rrcf.score_one(x)
    after = pickle.dumps(rrcf)
    assert before == after


def test_reproducibility():
    stream = _synthetic_stream(seed=7)

    def run():
        model = anomaly.RobustRandomCutForest(n_trees=15, tree_size=100, seed=42)
        out = []
        for x, _ in stream:
            out.append(model.score_one(x))
            model.learn_one(x)
        return out

    assert run() == run()


def test_duplicate_points():
    rrcf = anomaly.RobustRandomCutForest(n_trees=5, tree_size=32, seed=11)
    for _ in range(10):
        rrcf.learn_one({"a": 1.0, "b": 2.0})
        rrcf.learn_one({"a": -1.0, "b": 0.5})
    score = rrcf.score_one({"a": 1.0, "b": 2.0})
    assert score >= 0.0
    assert rrcf.score_one({"a": 50.0, "b": -50.0}) > score


def test_constant_stream():
    rrcf = anomaly.RobustRandomCutForest(n_trees=5, tree_size=16, seed=2)
    for _ in range(50):
        rrcf.learn_one({"a": 3.14})
    assert rrcf.score_one({"a": 3.14}) == 0.0
    assert rrcf.score_one({"a": 100.0}) >= 0.0


def test_leaf_split_cut_is_drawn_in_proportion_to_feature_spans():
    forest = anomaly.RobustRandomCutForest(n_trees=300, tree_size=8, seed=0)
    forest.learn_one({"a": 0.0, "b": 0.0})
    forest.learn_one({"a": 1.0, "b": 10.0})
    picks = collections.Counter(member.root.feature for member in forest.trees)
    assert picks["b"] > 5 * picks["a"]


def test_window_slides():
    tree_size = 16
    rrcf = anomaly.RobustRandomCutForest(n_trees=3, tree_size=tree_size, seed=5)
    rng = random.Random(0)
    for _ in range(5 * tree_size):
        rrcf.learn_one({"a": rng.gauss(0.0, 1.0), "b": rng.gauss(0.0, 1.0)})
    for member in rrcf.trees:
        assert len(member.leaves) == tree_size
        assert member.root.n_points == tree_size


def test_missing_and_extra_features_do_not_crash():
    rrcf = anomaly.RobustRandomCutForest(n_trees=5, tree_size=32, seed=0)
    rrcf.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    rrcf.learn_one({"a": 1.5})
    rrcf.score_one({"a": 1.0, "z": 9.0})
    rrcf.learn_one({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert isinstance(rrcf.score_one({"b": 2.0}), float)
