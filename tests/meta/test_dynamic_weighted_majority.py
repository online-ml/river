from skmultiflow.meta import DynamicWeightedMajority
from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
import numpy as np


def test_dynamic_weighted_majority():
    stream = SEAGenerator(1, noise_percentage=0.067, random_state=112)
    stream.prepare_for_use()

    learner = DynamicWeightedMajority(3, NaiveBayes(), beta=0.5, theta=0.01)

    cnt = 0
    max_samples = 5000
    predictions = []
    wait_samples = 100
    correct_predictions = 0
    first = True

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
    expected_predictions = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1,
                            0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
                            0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    expected_correct_predictions = 44
    expected_performance = 0.8979591836734694

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray

    expected_info = 'DynamicWeightedMajority(base_estimator=NaiveBayes(nominal_attributes=None),\n' \
                    '                        beta=0.5, n_estimators=3, period=50, theta=0.01)'
    assert learner.get_info() == expected_info
