from __future__ import annotations

import math

import pytest
from sklearn import metrics as sk_metrics

from river import metrics


def _feed(metric, y_true, y_pred, sample_weight=None):
    if sample_weight is None:
        for yt, yp in zip(y_true, y_pred):
            metric.update(yt, yp)
    else:
        for yt, yp, w in zip(y_true, y_pred, sample_weight):
            metric.update(yt, yp, w)
    return metric


def _scores(pos_val, y_true, y_pred, sample_weight=None):
    kwargs = {"pos_label": pos_val, "zero_division": 0}
    weight = {} if sample_weight is None else {"sample_weight": sample_weight}
    return {
        "precision": sk_metrics.precision_score(y_true, y_pred, **kwargs, **weight),
        "recall": sk_metrics.recall_score(y_true, y_pred, **kwargs, **weight),
        "f1": sk_metrics.f1_score(y_true, y_pred, **kwargs, **weight),
        "fbeta": sk_metrics.fbeta_score(y_true, y_pred, beta=0.5, **kwargs, **weight),
        "jaccard": sk_metrics.jaccard_score(y_true, y_pred, **kwargs, **weight),
        "mcc": sk_metrics.matthews_corrcoef(y_true, y_pred, **weight),
    }


CASES = [
    (
        "bool_docstring",
        True,
        [True, False, True, True, True],
        [True, True, False, True, True],
    ),
    ("perfect_one", 1, [1, 0, 1, 0], [1, 0, 1, 0]),
    (
        "always_wrong_bool",
        True,
        [True, False, True, False],
        [False, True, False, True],
    ),
    (
        "perfect_spam",
        "spam",
        ["spam", "ham", "spam", "ham"],
        ["spam", "ham", "spam", "ham"],
    ),
    (
        "always_wrong_spam",
        "spam",
        ["spam", "ham", "spam", "ham"],
        ["ham", "spam", "ham", "spam"],
    ),
    ("always_predict_zero", 0, [0, 0, 0, 1], [0, 0, 0, 0]),
    ("never_predict_zero", 0, [0, 1, 1, 1], [1, 1, 1, 1]),
    ("never_predict_false", False, [False, True, True], [True, True, True]),
    ("positive_is_false", False, [False, True, False, True], [False, False, True, True]),
]


@pytest.mark.parametrize(
    "name, pos_val, y_true, y_pred",
    [pytest.param(*case, id=case[0]) for case in CASES],
)
def test_binary_metrics_match_sklearn(name, pos_val, y_true, y_pred):
    expected = _scores(pos_val, y_true, y_pred)
    got = {
        "precision": _feed(metrics.Precision(pos_val=pos_val), y_true, y_pred).get(),
        "recall": _feed(metrics.Recall(pos_val=pos_val), y_true, y_pred).get(),
        "f1": _feed(metrics.F1(pos_val=pos_val), y_true, y_pred).get(),
        "fbeta": _feed(metrics.FBeta(beta=0.5, pos_val=pos_val), y_true, y_pred).get(),
        "jaccard": _feed(metrics.Jaccard(pos_val=pos_val), y_true, y_pred).get(),
        "mcc": _feed(metrics.MCC(pos_val=pos_val), y_true, y_pred).get(),
    }
    for key, value in expected.items():
        assert got[key] == pytest.approx(value), key


def test_sample_weight_and_shared_confusion_matrix():
    y_true = ["spam", "ham", "spam", "ham"]
    y_pred = ["spam", "spam", "ham", "ham"]
    sample_weight = [3.0, 1.0, 2.0, 4.0]
    expected = _scores("spam", y_true, y_pred, sample_weight)

    cm = metrics.ConfusionMatrix()
    f1 = metrics.F1(cm=cm, pos_val="spam")
    _feed(f1, y_true, y_pred, sample_weight)

    assert set(cm.classes) == {"spam", "ham"}
    assert f1.precision.get() == pytest.approx(expected["precision"])
    assert f1.recall.get() == pytest.approx(expected["recall"])
    assert f1.get() == pytest.approx(expected["f1"])
    assert _feed(
        metrics.Jaccard(pos_val="spam"), y_true, y_pred, sample_weight
    ).get() == pytest.approx(expected["jaccard"])
    assert _feed(
        metrics.MCC(pos_val="spam"), y_true, y_pred, sample_weight
    ).get() == pytest.approx(expected["mcc"])


def test_update_revert_round_trip():
    metric = metrics.Precision(pos_val="spam") + metrics.MCC(pos_val="spam")
    pairs = [("spam", "spam", 2.0), ("ham", "spam", 1.0), ("spam", "ham", 4.0)]
    for yt, yp, w in pairs:
        metric.update(yt, yp, w)
    assert metric[0].get() == pytest.approx(2.0 / 3.0)
    for yt, yp, w in reversed(pairs):
        metric.revert(yt, yp, w)
    assert metric[0].get() == pytest.approx(0.0)
    assert metric[1].get() == pytest.approx(0.0)
    assert metric[0].cm.total_weight == pytest.approx(0.0)


def test_mcc_counts_predictions_outside_the_two_labels_as_negatives():
    y_true = [True, False, False, True]
    y_pred = [True, None, False, False]
    metric = _feed(metrics.MCC(), y_true, y_pred)
    assert metric.get() == pytest.approx(1 / math.sqrt(3))


def test_roc_auc_uses_pos_val_probability():
    y_true = [0, 0, 1, 1]
    y_pred = [
        {0: 0.9, 1: 0.1},
        {0: 0.6, 1: 0.4},
        {0: 0.35, 1: 0.65},
        {0: 0.2, 1: 0.8},
    ]
    scores = [row[0] for row in y_pred]
    labels = [yt == 0 for yt in y_true]

    reference = metrics.ROCAUC(n_thresholds=20)
    reference_proba = metrics.ROCAUC(n_thresholds=20, pos_val=0)
    candidate = metrics.ROCAUC(n_thresholds=20, pos_val=0)
    for label, score, yt, yp in zip(labels, scores, y_true, y_pred):
        reference.update(label, score)
        reference_proba.update(yt, score)
        candidate.update(yt, yp)

    assert candidate.get() == pytest.approx(reference.get())
    assert candidate.get() == pytest.approx(reference_proba.get())
    assert candidate.get() > 0.5

    for yt, yp in zip(y_true, y_pred):
        candidate.revert(yt, yp)
    assert candidate.get() == pytest.approx(0.0)
