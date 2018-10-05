from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
import numpy as np


def test_adaptive_random_forests():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

    cnt = 0
    max_samples = 5000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(int(learner.predict(X)[0]))
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1

    performance = correct_predictions / len(predictions)
    expected_predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                            1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                            1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

    expected_correct_predictions = 32
    expected_performance = expected_correct_predictions / len(predictions)

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions


def test_adaptive_random_forests_labels_given():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112, n_classes=2)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y, classes=[0, 1])

    cnt = 0
    max_samples = 5000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict_proba(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1].argmax()):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1
    
    assert np.alltrue([np.isclose(y_proba.sum(), 1) for y_proba in predictions])

    performance = correct_predictions / len(predictions)
    expected_predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                            1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                            1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

    expected_correct_predictions = 32
    expected_performance = expected_correct_predictions / len(predictions)

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions