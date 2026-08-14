from __future__ import annotations

import math

from river import anomaly, linear_model, preprocessing


def test_learn_one_maintains_threshold_statistics():
    """The dynamic threshold statistics are maintained by `learn_one`, not `score_one`.

    `score_one` used to update the dynamic MAE/variance backing the threshold, so scoring a
    point changed the model (cf. the analogous issue #1331 for `anomaly.LocalOutlierFactor`).
    That `score_one` is now side-effect free is enforced globally by
    `checks.anomaly.check_score_one_does_not_mutate`; here we check the flip side, that
    `learn_one` takes over maintaining those statistics.
    """
    model = anomaly.PredictiveAnomalyDetection(
        predictive_model=preprocessing.StandardScaler() | linear_model.LinearRegression(),
        warmup_period=0,
    )
    assert model.dynamic_mae.n == 0

    for i in range(10):
        model.learn_one({"x": float(i)}, 2.0 * i + 1.0)

    # Every learned observation past the warm-up period contributes one error sample.
    assert model.dynamic_mae.n == 10
    assert model.dynamic_se_variance.n == 10


def test_score_one_before_any_learn_does_not_crash():
    """The default predictive model cannot predict before it has learnt anything.

    `preprocessing.MinMaxScaler` maps the very first observation to `0 / 0`, so the prediction,
    and hence the squared error, is NaN. Scoring used to fall through to `squared_error /
    threshold` with a threshold of zero and raise `ZeroDivisionError`, and the NaN also poisoned
    the dynamic statistics permanently once it reached them.
    """
    model = anomaly.PredictiveAnomalyDetection()
    x = {"a": 1.0, "b": 2.0}

    assert model.score_one(x, 1.0) == 0.0

    model.learn_one(x, 1.0)
    assert model.dynamic_mae.n == 0
    assert math.isfinite(model.score_one({"a": 2.0, "b": 3.0}, 2.0))

    for i in range(20):
        model.learn_one({"a": float(i), "b": float(i) + 1.0}, float(i))
    assert model.dynamic_mae.n > 0
    assert math.isfinite(model.dynamic_mae.get())
    assert math.isfinite(model.score_one({"a": 99.0, "b": 100.0}, 99.0))
