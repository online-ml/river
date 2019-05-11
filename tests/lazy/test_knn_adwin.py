from array import array
from skmultiflow.lazy import KNNAdwin
from skmultiflow.data import ConceptDriftStream, SEAGenerator
import numpy as np


def test_knn_adwin():
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2),
                                random_state=1, position=250, width=10)
    stream.prepare_for_use()
    learner = KNNAdwin(n_neighbors=8, leaf_size=40, max_window_size=200)

    cnt = 0
    max_samples = 1000
    predictions = array('i')
    correct_predictions = 0
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                                       0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       0, 1, 1, 1, 0, 1, 0, 1, 1])
    assert np.alltrue(predictions == expected_predictions)

    expected_correct_predictions = 46
    assert correct_predictions == expected_correct_predictions

    learner.reset()
    assert learner.window.n_samples == 0

    expected_info = 'KNNAdwin(leaf_size=40, max_window_size=200, n_neighbors=8,\n' \
                    '         nominal_attributes=None)'
    assert learner.get_info() == expected_info

    stream.restart()

    X, y = stream.next_sample(max_samples)
    learner.fit(X[:950], y[:950])
    predictions = learner.predict(X[951:])

    correct_predictions = sum(np.array(predictions) == y[951:])
    expected_correct_predictions = 47
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
