from __future__ import annotations

from river import anomaly


def test_learn_one_maintains_threshold_statistics():
    """The dynamic threshold statistics are maintained by `learn_one`, not `score_one`.

    `score_one` used to update the dynamic MAE/variance backing the threshold, so scoring a
    point changed the model (cf. the analogous issue #1331 for `anomaly.LocalOutlierFactor`).
    That `score_one` is now side-effect free is enforced globally by
    `checks.anomaly.check_score_one_does_not_mutate`; here we check the flip side, that
    `learn_one` takes over maintaining those statistics.
    """
    model = anomaly.PredictiveAnomalyDetection(warmup_period=0)
    assert model.dynamic_mae.n == 0

    for i in range(10):
        model.learn_one({"x": float(i)}, 2.0 * i + 1.0)

    # Every learned observation past the warm-up period contributes one error sample.
    assert model.dynamic_mae.n == 10
    assert model.dynamic_se_variance.n == 10
