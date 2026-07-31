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


def check_score_one_does_not_mutate(anomaly_detector, dataset):
    """Scoring must not mutate the anomaly detector; only ``learn_one`` may.

    A point's anomaly score must not depend on how many times it has been
    scored, so ``score_one`` has to leave the model's state untouched. This
    guards every anomaly detector (supervised or not) against accidentally
    updating state inside ``score_one``.
    """

    import pickle

    supervised = anomaly_detector._supervised

    first = None
    for x, y in dataset:
        if first is None:
            first = (x, y)
        if supervised:
            anomaly_detector.learn_one(x, y)
        else:
            anomaly_detector.learn_one(x)

    x, y = first
    snapshot = pickle.dumps(anomaly_detector)
    if supervised:
        anomaly_detector.score_one(x, y)
    else:
        anomaly_detector.score_one(x)

    assert pickle.dumps(anomaly_detector) == snapshot, f"score_one mutated {anomaly_detector!r}"
