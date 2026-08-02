from __future__ import annotations

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


def test_missing_and_extra_features_do_not_crash():
    rrcf = anomaly.RobustRandomCutForest(n_trees=5, tree_size=32, seed=0)
    rrcf.learn_one({"a": 1.0, "b": 2.0, "c": 3.0})
    rrcf.learn_one({"a": 1.5})
    rrcf.score_one({"a": 1.0, "z": 9.0})
    rrcf.learn_one({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
    assert isinstance(rrcf.score_one({"b": 2.0}), float)
