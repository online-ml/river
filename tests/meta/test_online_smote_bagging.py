from skmultiflow.meta import OnlineSMOTEBagging
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import SEAGenerator
import numpy as np


def test_online_smote_bagging():
    stream = SEAGenerator(1, noise_percentage=0.067, random_state=112)
    stream.prepare_for_use()
    nb = NaiveBayes()
    learner = OnlineSMOTEBagging(base_estimator=nb, n_estimators=3, sampling_rate=2, random_state=112)
    first = True

    cnt = 0
    max_samples = 5000
    predictions = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        if first:
            learner.partial_fit(X, y, classes=stream.target_values)
            first = False
        else:
            learner.partial_fit(X, y)
        cnt += 1
    performance = correct_predictions / len(predictions)
    expected_predictions = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                            0, 0, 1, 0, 1, 1, 1, 0, 1, 1,
                            0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                            0, 1, 1, 0, 1, 0, 0, 1, 1]

    expected_correct_predictions = 42
    expected_performance = 0.8571428571428571

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    expected_info = "OnlineSMOTEBagging(base_estimator=NaiveBayes(nominal_attributes=None),\n" \
                    "                   drift_detection=True, n_estimators=3, random_state=None,\n" \
                    "                   sampling_rate=2)"
    assert learner.get_info() == expected_info
