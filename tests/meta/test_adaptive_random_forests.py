from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
import numpy as np


def test_adaptive_random_forests():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

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
            predictions.append(int(learner.predict(X)[0]))
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1

        last_version_predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                            1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                            1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0]

    # Performance below does not need to be guaranteed. This check is set up so that anything that changes
    # to predictions are caught in the unit test. This helps prevent accidental changes.
    # If these tests fail, make sure that what is worked on *should* change the predictions of ARF.
    assert np.alltrue(predictions == last_version_predictions)


def test_adaptive_random_forests_labels_given():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112, n_classes=2)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y, classes=[0, 1])

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
            predictions.append(learner.predict_proba(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1].argmax()):
                correct_predictions += 1

        learner.partial_fit(X, y)
        cnt += 1
    
    assert np.alltrue([np.isclose(y_proba.sum(), 1) for y_proba in predictions]), "Probabilities should sum to 1."

    class_probabilities = np.asarray(predictions).squeeze()
    assert class_probabilities.shape == (49, 2)

    predictions = class_probabilities.argmax(axis=1)
    last_version_predictions = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0]
    # See comment in test `test_adaptive_random_forests`
    assert np.alltrue(predictions == last_version_predictions)


def test_adaptive_random_forests_batch_predict_proba():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112, n_classes=2)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y, classes=[0, 1])

    cnt = 0
    max_samples = 500
    predictions = []
    true_labels = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample(5)
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            p = learner.predict_proba(X)
            assert p.shape == (5, 2)
            predictions.append(p)
            true_labels.append(y)
        learner.partial_fit(X, y)
        cnt += 1

    all_predictions = np.concatenate(predictions)
    # all_true_labels = np.asarray(true_labels).flatten()
    # correct_predictions = sum(np.equal(all_true_labels, all_predictions.argmax(axis=1)))

    assert np.alltrue([np.isclose(y_proba.sum(), 1) for y_proba in all_predictions]), "Probabilities should sum to 1."
    assert all_predictions.shape == (4*5, 2)

    last_version_predictions = [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    # See comment in test `test_adaptive_random_forests`
    assert np.alltrue(all_predictions.argmax(axis=1) == last_version_predictions)
