from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.data import SEAGenerator, RandomTreeGenerator, ConceptDriftStream
from skmultiflow.bayes import NaiveBayes

import numpy as np
import pytest


def test_leverage_bagging():
    stream = SEAGenerator(classification_function=1,
                          noise_percentage=0.067,
                          random_state=112)
    knn = KNNClassifier(n_neighbors=8,
                        leaf_size=40,
                        max_window_size=2000)
    learner = LeveragingBaggingClassifier(base_estimator=knn,
                                          n_estimators=3,
                                          random_state=112)
    first = True

    cnt = 0
    max_samples = 5000
    predictions = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        if first:
            learner.partial_fit(X, y, classes=stream.target_values)
            first = False
        else:
            learner.partial_fit(X, y)
        cnt += 1

    performance = correct_predictions / len(predictions)
    expected_predictions = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                            0, 0, 1, 0, 1, 1, 1, 0, 1, 0,
                            0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
                            1, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                            0, 1, 1, 0, 1, 0, 0, 1, 1]
    assert np.alltrue(predictions == expected_predictions)

    expected_performance = 0.8571428571428571
    assert np.isclose(expected_performance, performance)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    expected_info = "LeveragingBaggingClassifier(base_estimator=KNNClassifier(leaf_size=40, " \
                    "max_window_size=2000, metric='euclidean', n_neighbors=8), " \
                    "delta=0.002, enable_code_matrix=False, leverage_algorithm='leveraging_bag'," \
                    " n_estimators=3, random_state=112, w=6)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_leverage_bagging_me():
    stream = ConceptDriftStream(position=500,
                                width=100,
                                random_state=112)
    nb = NaiveBayes()

    # leveraging_bag_me
    learner = LeveragingBaggingClassifier(base_estimator=nb,
                                          n_estimators=5,
                                          random_state=112,
                                          leverage_algorithm='leveraging_bag_me')

    y_expected = np.asarray([0, 0, 0, 1, 0, 1, 0, 0, 1, 0,
                             0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                             1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
                             0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
                             1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40,
                               y_expected=y_expected)


def test_leverage_bagging_half():
    stream = SEAGenerator(classification_function=1,
                          noise_percentage=0.067,
                          random_state=112)
    knn = KNNClassifier(n_neighbors=8,
                        leaf_size=40,
                        max_window_size=2000)
    # leveraging_bag_half
    learner = LeveragingBaggingClassifier(base_estimator=knn,
                                          n_estimators=3,
                                          random_state=112,
                                          leverage_algorithm='leveraging_bag_half')

    y_expected = np.asarray([0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                             1, 0, 0, 0, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
                             0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                             1, 0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40,
                               y_expected=y_expected)


def test_leverage_bagging_wt():
    stream = SEAGenerator(classification_function=1,
                          noise_percentage=0.067,
                          random_state=112)
    knn = KNNClassifier(n_neighbors=8,
                        leaf_size=40,
                        max_window_size=2000)

    # leveraging_bag_wt
    learner = LeveragingBaggingClassifier(base_estimator=knn,
                                          n_estimators=3,
                                          random_state=112,
                                          leverage_algorithm='leveraging_bag_wt')

    y_expected = np.asarray([0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                             1, 0, 0, 0, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
                             0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                             1, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40,
                               y_expected=y_expected)


def test_leverage_bagging_subag():
    stream = SEAGenerator(classification_function=1,
                          noise_percentage=0.067,
                          random_state=112)
    knn = KNNClassifier(n_neighbors=8,
                        leaf_size=40,
                        max_window_size=2000)

    # leveraging_subag
    learner = LeveragingBaggingClassifier(base_estimator=knn,
                                          n_estimators=3,
                                          random_state=112,
                                          leverage_algorithm='leveraging_subag')

    y_expected = np.asarray([0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                             1, 1, 0, 0, 1, 0, 1, 1, 1, 1,
                             1, 1, 1, 1, 0, 1, 0, 1, 1, 0,
                             0, 0, 1, 1, 1, 0, 1, 1, 0, 0,
                             1, 0, 0, 1, 0, 0, 0, 1, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40,
                               y_expected=y_expected)


def test_leverage_bagging_code_matrix():
    stream = RandomTreeGenerator(tree_random_state=1,
                                 sample_random_state=12,
                                 n_classes=5)
    nb = NaiveBayes()

    # enable the output detection code matrix
    learner = LeveragingBaggingClassifier(base_estimator=nb,
                                          n_estimators=5,
                                          random_state=12,
                                          enable_code_matrix=True)

    y_expected = np.asarray([0, 0, 3, 2, 3, 1, 4, 1, 3, 4,
                             2, 4, 2, 2, 0, 0, 2, 4, 2, 4,
                             0, 4, 2, 4, 2, 4, 0, 4, 1, 3,
                             2, 1, 2, 4, 2, 4, 1, 3, 0, 4,
                             2, 0, 0, 4, 3, 2, 4, 4, 2, 4], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40,
                               y_expected=y_expected)


def test_leverage_bagging_coverage():
    # Invalid leverage_algorithm
    with pytest.raises(ValueError):
        LeveragingBaggingClassifier(leverage_algorithm='invalid')

    estimator = LeveragingBaggingClassifier(random_state=4321)
    stream = SEAGenerator(random_state=4321)
    X, y = stream.next_sample()

    # classes not passed in partial_fit
    with pytest.raises(ValueError):
        estimator.partial_fit(X, y, classes=None)
    estimator.partial_fit(X, y, classes=stream.target_values)
    # different observed classes
    with pytest.raises(ValueError):
        estimator.partial_fit(X, y, classes=stream.target_values + [-1])
    # Invalid leverage_algorithm, changed after initialization
    with pytest.raises(RuntimeError):
        estimator.leverage_algorithm = 'invalid'
        estimator.partial_fit(X, y, classes=stream.target_values)

    # Reset ensemble
    estimator.reset()
    assert estimator.classes is None


def run_prequential_supervised(stream, learner, max_samples, n_wait, y_expected=None):
    stream.restart()

    y_pred = np.zeros(max_samples // n_wait, dtype=np.int)
    y_true = np.zeros(max_samples // n_wait, dtype=np.int)
    j = 0

    for i in range(max_samples):
        X, y = stream.next_sample()
        # Test every n samples
        if i % n_wait == 0:
            y_pred[j] = int(learner.predict(X)[0])
            y_true[j] = (y[0])
            j += 1
        learner.partial_fit(X, y, classes=stream.target_values)

    assert type(learner.predict(X)) == np.ndarray

    if y_expected is not None:
        assert np.alltrue(y_pred == y_expected)