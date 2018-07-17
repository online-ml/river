import numpy as np
from skmultiflow.trees import LCHT
from skmultiflow.data import MultilabelGenerator
import os

def test_lc_hoeffding_tree(test_path):
    stream = MultilabelGenerator(n_samples=10000, n_features=15, n_targets=3, n_labels=4, random_state=112)

    stream.prepare_for_use()

    learner = LCHT(n_labels=3)

    cnt = 0
    max_samples = 5000
    predictions = []
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        learner.partial_fit(X, y)
        if cnt % wait_samples == 0 and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        cnt += 1

    expected_predictions = [[0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1],
                           [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 1],
                           [0, 1, 1], [1, 1, 1], [0, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1],
                           [1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1], [0, 0, 0],
                           [0, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0], [1, 1, 1], [1, 1, 1],
                           [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1],
                           [1, 1, 1]]

    assert np.alltrue(predictions == expected_predictions)