from __future__ import annotations

import pickle

from river import anomaly


def _fit(model, n=30):
    """Feed a simple deterministic linear stream, scoring then learning each point."""
    for i in range(n):
        x = {"x": float(i)}
        y = 2.0 * i + 1.0
        model.score_one(x, y)
        model.learn_one(x, y)
    return model


def test_score_one_does_not_mutate_state():
    """`score_one` must be side-effect free.

    Regression test: `score_one` used to update the dynamic MAE/variance backing the threshold,
    so scoring a point changed the model and repeated scoring of the same point returned
    different values (cf. the analogous issue #1331 for `anomaly.LocalOutlierFactor`).
    """
    model = _fit(anomaly.PredictiveAnomalyDetection(warmup_period=0))

    mae_n = model.dynamic_mae.n
    var_n = model.dynamic_se_variance.n
    snapshot = pickle.dumps(model)

    probe_x, probe_y = {"x": 30.0}, 999.0
    scores = [model.score_one(probe_x, probe_y) for _ in range(5)]

    # Scoring left every statistic untouched...
    assert model.dynamic_mae.n == mae_n
    assert model.dynamic_se_variance.n == var_n
    assert pickle.dumps(model) == snapshot
    # ...so it is idempotent.
    assert len(set(scores)) == 1


def test_learn_one_maintains_threshold_statistics():
    """The dynamic threshold statistics are maintained by `learn_one`, not `score_one`."""
    model = anomaly.PredictiveAnomalyDetection(warmup_period=0)
    assert model.dynamic_mae.n == 0

    for i in range(10):
        model.learn_one({"x": float(i)}, 2.0 * i + 1.0)

    # Every learned observation past the warm-up period contributes one error sample.
    assert model.dynamic_mae.n == 10
    assert model.dynamic_se_variance.n == 10
