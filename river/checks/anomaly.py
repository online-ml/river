from __future__ import annotations


def check_roc_auc(anomaly_detector, dataset):
    """A detector should rank anomalies above normal points (ROC AUC >= 50%).

    Each sample is scored *before* it is learned (prequential evaluation), so the detector is
    never asked to score a point it has already memorised — which would leak the label and
    inflate the score.
    """

    from sklearn import metrics

    scores = []
    labels = []

    for x, y in dataset:
        scores.append(anomaly_detector.score_one(x))
        anomaly_detector.learn_one(x)
        labels.append(y)

    assert metrics.roc_auc_score(labels, scores) >= 0.5
