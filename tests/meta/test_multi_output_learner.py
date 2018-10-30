from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.data import MultilabelGenerator
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.metrics.measure_collection import hamming_score
import numpy as np


def test_multi_output_learner():

    stream = MultilabelGenerator(n_samples=5150, n_features=15, n_targets=3, n_labels=4, random_state=112)
    stream.prepare_for_use()

    classifier = MultiOutputLearner(base_estimator=HoeffdingTree())

    X, y = stream.next_sample(150)
    classifier.partial_fit(X, y)

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
            predictions.append(classifier.predict(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        classifier.partial_fit(X, y)
        cnt += 1

    perf = hamming_score(true_labels, predictions)
    expected_predictions = [[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [0., 1., 1.],
                            [1., 0., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.],
                            [0., 0., 1.],
                            [0., 1., 1.],
                            [0., 1., 1.],
                            [1., 1., 1.],
                            [0., 1., 1.],
                            [1., 1., 0.],
                            [1., 1., 1.],
                            [0., 1., 1.],
                            [1., 0., 0.],
                            [1., 0., 1.],
                            [1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 0., 0.],
                            [0., 1., 1.],
                            [1., 1., 0.],
                            [1., 1., 1.],
                            [0., 1., 1.],
                            [1., 1., 1.],
                            [0., 1., 1.],
                            [1., 0., 1.],
                            [1., 0., 1.],
                            [0., 0., 1.],
                            [0., 1., 1.],
                            [1., 1., 0.],
                            [0., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 0., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [0., 1., 1.]]
    expected_correct_predictions = 32

    expected_performance = 0.8503401360544217

    assert np.alltrue(np.array_equal(predictions, expected_predictions))
    assert np.isclose(expected_performance, perf)
    assert correct_predictions == expected_correct_predictions

    assert type(classifier.predict(X)) == np.ndarray
    assert type(classifier.predict_proba(X)) == np.ndarray
