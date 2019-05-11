import numpy as np
from array import array
import os
from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.core.base import is_classifier

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_naive_bayes(test_path):
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()

    learner = NaiveBayes()

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    X_batch = []
    y_batch = []
    y_proba = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y, classes=stream.target_values)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 1, 0, 0, 1,
                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                       0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
                                       1, 1, 0, 1, 0, 0, 1, 1, 1])

    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'data_naive_bayes_proba.npy')
    y_proba_expected = np.load(test_file)
    assert np.allclose(y_proba, y_proba_expected)

    expected_info = 'NaiveBayes(nominal_attributes=None)'
    assert learner.get_info() == expected_info

    learner.reset()
    learner.fit(X=np.array(X_batch[:4500]), y=np.array(y_batch[:4500]))

    expected_score = 0.9378757515030061
    assert np.isclose(expected_score, learner.score(X=np.array(X_batch[4501:]),
                                                    y=np.array(y_batch[4501:])))

    assert is_classifier(learner)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
