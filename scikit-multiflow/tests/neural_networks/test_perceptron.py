import numpy as np

from array import array
import os

from skmultiflow.data import SEAGenerator
from skmultiflow.neural_networks import PerceptronMask
from sklearn.metrics import accuracy_score

import pytest
from sklearn import __version__ as sklearn_version


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_perceptron(test_path):
    stream = SEAGenerator(random_state=1)

    learner = PerceptronMask(random_state=1)

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

    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
                                       0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
                                       1, 1, 0, 1, 0, 1, 1, 0, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'data_perceptron_proba.npy')
    y_proba_expected = np.load(test_file)
    assert np.allclose(y_proba, y_proba_expected)

    expected_info = "PerceptronMask(alpha=0.0001, class_weight=None, early_stopping=False, " \
                    "eta0=1.0, fit_intercept=True, max_iter=1000, n_iter_no_change=5, " \
                    "n_jobs=None, penalty=None, random_state=1, shuffle=True, tol=0.001, " \
                    "validation_fraction=0.1, verbose=0, warm_start=False)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    # Coverage tests
    learner.reset()
    if not sklearn_version.startswith("0.21"):
        learner.fit(X=np.asarray(X_batch[:4500]), y=np.asarray(y_batch[:4500], dtype=int))
    else:
        # Root cause of failure (TypeError: an integer is required) is in the fit() method
        # in sklearn 0.21.0. This is a workaround until a fix is made available in sklearn
        learner.partial_fit(X=np.asarray(X_batch[:4500]), y=np.asarray(y_batch[:4500]),
                            classes=stream.target_values)
    learner.predict(X=X_batch[4501:])   # Run for coverage

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
