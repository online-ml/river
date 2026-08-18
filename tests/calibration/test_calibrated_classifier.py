from __future__ import annotations

import math

from river import base, calibration, datasets, evaluate, linear_model, metrics, preprocessing, utils


def test_identity_initially() -> None:
    # With a=1 and b=0, the calibrated probabilities equal the wrapped ones.
    wrapped = linear_model.LogisticRegression()
    model = calibration.CalibratedClassifier(wrapped)
    x: dict[base.typing.FeatureName, float] = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}
    y_pred = wrapped.predict_proba_one(x)
    y_pred_cal = model.predict_proba_one(x)
    assert math.isclose(y_pred[True], y_pred_cal[True])
    assert math.isclose(y_pred[False], y_pred_cal[False])


def test_predict_proba_is_a_distribution() -> None:
    model = calibration.CalibratedClassifier(linear_model.LogisticRegression())
    for x, y in datasets.Phishing().take(300):
        model.learn_one(x, y)
        y_pred = model.predict_proba_one(x)
        assert set(y_pred.keys()) == {False, True}
        assert 0.0 <= y_pred[True] <= 1.0
        assert 0.0 <= y_pred[False] <= 1.0
        assert math.isclose(y_pred[True] + y_pred[False], 1.0)


def test_score_is_logit_of_wrapped_max_probability() -> None:
    # The score fed to the sigmoid is the logit of the wrapped classifier's maximum
    # probability, and predict_proba_one applies sigmoid(a * s + b) to it. That probability is
    # assigned to the label the wrapped model is most confident about.
    wrapped = linear_model.PAClassifier()
    cal = calibration.CalibratedClassifier(wrapped.clone())
    for x, y in datasets.Phishing().take(500):
        cal.learn_one(x, y)
        wrapped.learn_one(x, y)

    for x, _ in datasets.Phishing().take(200):
        label, p = max(wrapped.predict_proba_one(x).items(), key=lambda kv: kv[1])
        s = math.log(p / (1 - p))
        expected = utils.math.sigmoid(cal.a * s + cal.b)
        assert math.isclose(cal.predict_proba_one(x)[label], expected)


def test_calibration_improves_log_loss() -> None:
    # PAClassifier outputs hinge-margin values, not probabilities, so its log-loss is high and
    # Platt scaling on top of it should bring it down.
    dataset = datasets.Phishing().take(1000)

    wrapped = preprocessing.StandardScaler() | linear_model.PAClassifier()
    raw_loss = evaluate.progressive_val_score(dataset, wrapped, metrics.LogLoss())

    calibrated = calibration.CalibratedClassifier(
        preprocessing.StandardScaler() | linear_model.PAClassifier()
    )
    cal_loss = evaluate.progressive_val_score(dataset, calibrated, metrics.LogLoss())

    assert cal_loss.get() <= raw_loss.get()


def test_learns_non_trivial_params() -> None:
    model = calibration.CalibratedClassifier(linear_model.LogisticRegression())
    for x, y in datasets.Phishing().take(500):
        model.learn_one(x, y)
    # The model should have moved away from the identity mapping.
    assert not (math.isclose(model.a, 1.0) and math.isclose(model.b, 0.0))
