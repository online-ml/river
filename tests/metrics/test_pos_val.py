from __future__ import annotations

import math

import pytest
from sklearn import metrics as sk_metrics

from river import metrics


@pytest.mark.parametrize(
    ("pos_val", "y_true", "y_pred"),
    [
        (True, [True, False, True, True, True], [True, True, False, True, True]),
        (1, [1, 0, 1, 0], [1, 0, 1, 0]),
        (0, [0, 0, 0, 1], [0, 0, 0, 0]),
        (False, [False, True, True], [True, True, True]),
        ("spam", ["spam", "ham", "spam", "ham"], ["spam", "ham", "spam", "ham"]),
        ("spam", ["spam", "ham", "spam", "ham"], ["ham", "spam", "ham", "spam"]),
    ],
)
@pytest.mark.parametrize("beta", [1, 2])
def test_binary_metrics_match_one_vs_rest(pos_val, y_true, y_pred, beta):
    y_true_bin = [yt == pos_val for yt in y_true]
    y_pred_bin = [yp == pos_val for yp in y_pred]

    river_metrics = {
        "precision": metrics.Precision(pos_val=pos_val),
        "recall": metrics.Recall(pos_val=pos_val),
        "fbeta": metrics.FBeta(beta=beta, pos_val=pos_val),
        "jaccard": metrics.Jaccard(pos_val=pos_val),
        "mcc": metrics.MCC(pos_val=pos_val),
    }
    for yt, yp in zip(y_true, y_pred):
        for metric in river_metrics.values():
            metric.update(yt, yp)

    expected = {
        "precision": sk_metrics.precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": sk_metrics.recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "fbeta": sk_metrics.fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0),
        "jaccard": sk_metrics.jaccard_score(y_true_bin, y_pred_bin, zero_division=0),
        "mcc": sk_metrics.matthews_corrcoef(y_true_bin, y_pred_bin),
    }
    for name, metric in river_metrics.items():
        assert math.isclose(metric.get(), expected[name], abs_tol=1e-9), name


@pytest.mark.parametrize(
    ("pos_val", "y_true", "y_pred", "w"),
    [
        (0, [0, 1, 0, 1], [0, 0, 0, 1], [1, 2, 3, 4]),
        (
            "spam",
            ["spam", "ham", "spam", "ham"],
            ["spam", "spam", "ham", "ham"],
            [1, 2, 3, 4],
        ),
    ],
)
def test_binary_metrics_match_weighted_one_vs_rest(pos_val, y_true, y_pred, w):
    y_true_bin = [yt == pos_val for yt in y_true]
    y_pred_bin = [yp == pos_val for yp in y_pred]

    river_metrics = {
        "precision": metrics.Precision(pos_val=pos_val),
        "recall": metrics.Recall(pos_val=pos_val),
        "fbeta": metrics.FBeta(beta=2, pos_val=pos_val),
        "jaccard": metrics.Jaccard(pos_val=pos_val),
        "mcc": metrics.MCC(pos_val=pos_val),
    }
    for yt, yp, weight in zip(y_true, y_pred, w):
        for metric in river_metrics.values():
            metric.update(yt, yp, weight)

    expected = {
        "precision": sk_metrics.precision_score(
            y_true_bin, y_pred_bin, sample_weight=w, zero_division=0
        ),
        "recall": sk_metrics.recall_score(y_true_bin, y_pred_bin, sample_weight=w, zero_division=0),
        "fbeta": sk_metrics.fbeta_score(
            y_true_bin, y_pred_bin, beta=2, sample_weight=w, zero_division=0
        ),
        "jaccard": sk_metrics.jaccard_score(
            y_true_bin, y_pred_bin, sample_weight=w, zero_division=0
        ),
        "mcc": sk_metrics.matthews_corrcoef(y_true_bin, y_pred_bin, sample_weight=w),
    }
    for name, metric in river_metrics.items():
        assert math.isclose(metric.get(), expected[name], abs_tol=1e-9), name


@pytest.mark.parametrize(
    ("pos_val", "y_true", "y_pred"),
    [
        (0, [0, 1, 0, 1], [0, 0, 0, 1]),
        ("spam", ["spam", "ham", "spam", "ham"], ["spam", "ham", "spam", "ham"]),
    ],
)
def test_binary_metrics_revert_round_trip(pos_val, y_true, y_pred):
    cm = metrics.ConfusionMatrix()
    metric = metrics.FBeta(beta=2, cm=cm, pos_val=pos_val)

    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    value = metric.get()
    for yt, yp in zip(y_true, y_pred):
        metric.revert(yt, yp)

    assert cm.total_weight == 0
    assert metric.get() == 0
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    assert math.isclose(metric.get(), value)


def test_mcc_one_vs_rest_with_prediction_outside_labels():
    y_true = [True, False, False, True]
    y_pred = [True, None, False, False]

    metric = metrics.MCC()
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)

    expected = sk_metrics.matthews_corrcoef(y_true, [yp is True for yp in y_pred])
    assert math.isclose(metric.get(), expected, abs_tol=1e-9)


def test_classification_report_keeps_raw_label_counts():
    y_true = ["pear", "apple", "banana", "banana", "banana"]
    y_pred = ["apple", "pear", "banana", "banana", "apple"]

    report = metrics.ClassificationReport()
    for yt, yp in zip(y_true, y_pred):
        report.update(yt, yp)

    expected = sk_metrics.f1_score(
        [yt == "banana" for yt in y_true], [yp == "banana" for yp in y_pred]
    )
    assert math.isclose(metrics.F1(cm=report.cm, pos_val="banana").get(), expected)
    assert "100.00%" in repr(report)


@pytest.mark.parametrize(
    ("pos_val", "y_true", "scores"),
    [
        (True, [True, False, True, False], [0.9, 0.1, 0.8, 0.2]),
        (0, [0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2]),
        ("spam", ["spam", "ham", "spam", "ham"], [0.9, 0.1, 0.8, 0.2]),
    ],
)
def test_roc_auc_uses_positive_class_probability(pos_val, y_true, scores):
    neg_val = next(label for label in y_true if label != pos_val)
    metric = metrics.ROCAUC(n_thresholds=20, pos_val=pos_val)

    for yt, score in zip(y_true, scores):
        metric.update(yt, {pos_val: score, neg_val: 1 - score})

    expected = sk_metrics.roc_auc_score([yt == pos_val for yt in y_true], scores)
    assert math.isclose(metric.get(), expected, abs_tol=0.05)
