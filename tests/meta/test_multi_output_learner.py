from sklearn.linear_model import SGDClassifier
from sklearn import __version__ as sklearn_version

from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.data import MultilabelGenerator
from skmultiflow.metrics.measure_collection import hamming_score

import numpy as np

import pytest


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_multi_output_learner():

    stream = MultilabelGenerator(n_samples=5150, n_features=15, n_targets=3, n_labels=4, random_state=112)
    stream.prepare_for_use()

    estimator = SGDClassifier(random_state=112, tol=1e-3, max_iter=10, loss='log')
    classifier = MultiOutputLearner(base_estimator=estimator)

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

    if not sklearn_version.startswith("0.21"):
        expected_predictions = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0],
                                [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        assert np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 26
        assert correct_predictions == expected_correct_predictions

        expected_performance = 0.7755102040816326
        performance = hamming_score(true_labels, predictions)
        assert np.isclose(performance, expected_performance)

        expected_info = "MultiOutputLearner(base_estimator=SGDClassifier(alpha=0.0001, average=False, " \
                        "class_weight=None,\n" \
                        "       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,\n" \
                        "       l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=10,\n" \
                        "       n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',\n" \
                        "       power_t=0.5, random_state=112, shuffle=True, tol=0.001,\n" \
                        "       validation_fraction=0.1, verbose=0, warm_start=False))"
        assert classifier.get_info() == expected_info

    else:
        expected_predictions = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0],
                              [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                              [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0],
                              [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                              [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 23
        assert correct_predictions == expected_correct_predictions

        expected_performance = 0.7482993197278911
        performance = hamming_score(true_labels, predictions)
        assert np.isclose(performance, expected_performance)

        expected_info = "MultiOutputLearner(base_estimator=SGDClassifier(alpha=0.0001, average=False, " \
                        "class_weight=None,\n" \
                        "              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,\n" \
                        "              l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=10,\n" \
                        "              n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,\n" \
                        "              random_state=112, shuffle=True, tol=0.001,\n" \
                        "              validation_fraction=0.1, verbose=0, warm_start=False))"

        assert classifier.get_info() == expected_info

    assert type(classifier.predict(X)) == np.ndarray
    assert type(classifier.predict_proba(X)) == np.ndarray






