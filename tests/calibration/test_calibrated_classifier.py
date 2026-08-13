from __future__ import annotations

import math

from river import calibration, datasets, evaluate, linear_model, metrics, preprocessing


def test_identity_initially() -> None:
    # With a=1 and b=0, the calibrated probabilities equal the wrapped ones.
    wrapped = linear_model.LogisticRegression()
    model = calibration.CalibratedClassifier(wrapped)
    x = {i: 0.1 * i for i in range(5)}
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


def test_predict_one_matches_wrapped_label() -> None:
    # The sigmoid is monotonic, so calibration never changes the predicted label.
    wrapped = linear_model.PAClassifier()
    cal = calibration.CalibratedClassifier(wrapped.clone())
    for x, y in datasets.Phishing().take(500):
        assert cal.predict_one(x) == wrapped.predict_one(x)
        cal.learn_one(x, y)
        wrapped.learn_one(x, y)


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