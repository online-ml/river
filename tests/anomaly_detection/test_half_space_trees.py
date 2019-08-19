import os
from array import array

import numpy as np

from skmultiflow.anomaly_detection import HalfSpaceTrees
from skmultiflow.data import SEAGenerator


def test_half_space_trees(test_path):
    stream = SEAGenerator(classification_function=0, noise_percentage=0.1, random_state=1)
    stream.prepare_for_use()
    learner = HalfSpaceTrees(n_features=stream.n_features, n_estimators=13, size_limit=75, anomaly_threshold=0.90,
                             depth=10, random_state=5)

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    y_proba = []
    wait_samples = 500

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Scale inputs between 0 and 1
        X = X / 10
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 0, 0, 1, 0, 0, 1, 0])
    assert np.alltrue(y_pred == expected_predictions)
    test_file = os.path.join(test_path, 'test_half_space_trees.npy')
    expected_proba = np.load(test_file)
    assert np.allclose(y_proba, expected_proba)
