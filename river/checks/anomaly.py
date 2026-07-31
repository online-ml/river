from __future__ import annotations


def check_roc_auc(anomaly_detector, dataset):
    """A detector should rank anomalies above normal points (ROC AUC >= 50%).

    Each sample is scored *before* it is learned (prequential evaluation), so the detector is
    never asked to score a point it has already memorised, which would leak the label and
    inflate the score.
    """

    from sklearn import metrics

    supervised = anomaly_detector._supervised
    scores = []
    labels = []

    for x, y in dataset:
        if supervised:
            scores.append(anomaly_detector.score_one(x, y))
            anomaly_detector.learn_one(x, y)
        else:
            scores.append(anomaly_detector.score_one(x))
            anomaly_detector.learn_one(x)
        labels.append(y)

    assert metrics.roc_auc_score(labels, scores) >= 0.5


def check_score_one_does_not_mutate(anomaly_detector, dataset):
    """Scoring must not mutate the anomaly detector; only ``learn_one`` may.

    A point's anomaly score must not depend on how many times it has been scored, so
    ``score_one`` has to leave the model's state untouched. This is checked at every step of the
    stream, interleaved with ``learn_one``, which is stricter than checking only once at the end.
    """

    import pickle

    supervised = anomaly_detector._supervised

    def score(x, y):
        return anomaly_detector.score_one(x, y) if supervised else anomaly_detector.score_one(x)

    def learn(x, y):
        if supervised:
            anomaly_detector.learn_one(x, y)
        else:
            anomaly_detector.learn_one(x)

    # A first score warms any lazy prediction cache (memoisation, not learning), so the per-step
    # check below targets real state changes rather than one-off cache initialisation.
    x0, y0 = dataset[0]
    score(x0, y0)

    for x, y in dataset:
        snapshot = pickle.dumps(anomaly_detector)
        score(x, y)
        assert pickle.dumps(anomaly_detector) == snapshot, f"score_one mutated {anomaly_detector!r}"
        learn(x, y)
