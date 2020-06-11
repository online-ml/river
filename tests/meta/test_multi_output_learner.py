from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import __version__ as sklearn_version
from sklearn import set_config

from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.data import MultilabelGenerator, RegressionGenerator
from skmultiflow.metrics.measure_collection import hamming_score

import numpy as np

import pytest

from distutils.version import LooseVersion

# Force sklearn to show only the parameters whose default value have been changed when
# printing an estimator (backwards compatibility with versions prior to sklearn==0.23)
set_config(print_changed_only=True)


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_multi_output_learner_classifier():

    stream = MultilabelGenerator(n_samples=5150,
                                 n_features=15,
                                 n_targets=3,
                                 n_labels=4,
                                 random_state=112)

    estimator = SGDClassifier(random_state=112, max_iter=10, loss='log')
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

    if LooseVersion(sklearn_version) < LooseVersion("0.21"):
        expected_predictions = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0],
                                [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0]]
        assert np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 26
        assert correct_predictions == expected_correct_predictions

        expected_performance = 0.7755102040816326
        performance = hamming_score(true_labels, predictions)
        assert np.isclose(performance, expected_performance)

        expected_info = "MultiOutputLearner(base_estimator=SGDClassifier(loss='log', " \
                        "random_state=112))"
        info = " ".join([line.strip() for line in classifier.get_info().split()])
        assert info == expected_info

    else:
        expected_predictions = [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                                [1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0],
                                [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
                                [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0],
                                [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
                                [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0],
                                [0.0, 1.0, 1.0]]
        np.alltrue(np.array_equal(predictions, expected_predictions))

        expected_correct_predictions = 23
        assert correct_predictions == expected_correct_predictions

        expected_performance = 0.7482993197278911
        performance = hamming_score(true_labels, predictions)
        assert np.isclose(performance, expected_performance)

        expected_info = "MultiOutputLearner(base_estimator=SGDClassifier(loss='log', " \
                        "max_iter=10, random_state=112))"

        info = " ".join([line.strip() for line in classifier.get_info().split()])
        assert info == expected_info

    assert type(classifier.predict(X)) == np.ndarray
    assert type(classifier.predict_proba(X)) == np.ndarray


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_multi_output_learner_regressor():

    stream = RegressionGenerator(n_samples=5500,
                                 n_features=10,
                                 n_informative=20,
                                 n_targets=2,
                                 random_state=1)

    estimator = SGDRegressor(random_state=112, tol=1e-3, max_iter=10, loss='squared_loss')
    learner = MultiOutputLearner(base_estimator=estimator)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

    cnt = 0
    max_samples = 5000
    predictions = []
    true_targets = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            true_targets.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1

    expected_performance = 2.444365309339395
    performance = mean_absolute_error(true_targets, predictions)
    assert np.isclose(performance, expected_performance)

    assert learner._estimator_type == "regressor"
    assert type(learner.predict(X)) == np.ndarray

    with pytest.raises(AttributeError):
        learner.predict_proba(X)
