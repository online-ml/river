from __future__ import annotations


def check_roc_auc(anomaly_detector, dataset):
    """The ROC AUC should always be above 50%."""

    from sklearn import metrics

    scores = []
    labels = []

    for x, y in dataset:
        anomaly_detector.learn_one(x)
        y_pred = anomaly_detector.score_one(x)

        scores.append(y_pred)
        labels.append(y)

    assert metrics.roc_auc_score(labels, scores) >= 0.5
