from array import array
from skmultiflow.lazy.knn import KNN
from skmultiflow.data import SEAGenerator
import numpy as np


def test_knn():
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()

    learner = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
    cnt = 0
    max_samples = 5000
    predictions = array('i')
    correct_predictions = 0
    wait_samples = 100
    X_batch = []
    y_batch = []

    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                       0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
                                       1, 1, 0, 1, 0, 0, 1, 0, 1])
    assert np.alltrue(predictions == expected_predictions)

    expected_correct_predictions = 49
    assert correct_predictions == expected_correct_predictions

    expected_info = 'KNN(leaf_size=40, max_window_size=2000, n_neighbors=8, nominal_attributes=None)'
    assert learner.get_info() == expected_info

    learner.reset()
    assert learner.get_info() == expected_info

    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    learner.fit(X_batch[:4500], y_batch[:4500], classes=[0, 1])
    predictions = learner.predict(X_batch[4501:4550])

    expected_predictions = array('i', [1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                       1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                                       1, 0, 1, 1, 1, 1, 0, 0, 1, 0,
                                       0, 1, 1, 1, 0, 0, 1, 0, 0, 1,
                                       1, 1, 1, 1, 1, 1, 0, 1, 0])
    assert np.alltrue(predictions == expected_predictions)

    correct_predictions = sum(predictions == y_batch[4501:4550])
    expected_correct_predictions = 49
    assert correct_predictions == expected_correct_predictions

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
