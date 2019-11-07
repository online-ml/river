import numpy as np

from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest


def test_adaptive_random_forests_mc():
    stream = RandomTreeGenerator(
        tree_random_state=112, sample_random_state=112, n_classes=2
    )
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3, leaf_prediction='mc',
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
    last_version_predictions = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,
                                1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                                1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                                1]

    # Performance below does not need to be guaranteed. This check is set up so that anything that changes
    # to predictions are caught in the unit test. This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(predictions == last_version_predictions)

    expected_info = "AdaptiveRandomForest(binary_split=False, disable_weighted_vote=False,\n" \
                    "                     drift_detection_method=ADWIN(delta=0.001), grace_period=50,\n" \
                    "                     lambda_value=6, leaf_prediction='mc',\n" \
                    "                     max_byte_size=33554432, max_features=5,\n" \
                    "                     memory_estimate_period=2000000, n_estimators=3,\n" \
                    "                     nb_threshold=0, no_preprune=False, nominal_attributes=None,\n" \
                    "                     performance_metric='acc', random_state=112,\n" \
                    "                     remove_poor_atts=False, split_confidence=0.01,\n" \
                    "                     split_criterion='info_gain', stop_mem_management=False,\n" \
                    "                     tie_threshold=0.05,\n" \
                    "                     warning_detection_method=ADWIN(delta=0.01))"
    assert learner.get_info() == expected_info


def test_adaptive_random_forests_nb():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112, n_classes=2)
    stream.prepare_for_use()

    learner = AdaptiveRandomForest(n_estimators=3,
                                   random_state=112, leaf_prediction='nb')

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
    last_version_predictions = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
                                1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
                                1]

    # Performance below does not need to be guaranteed. This check is set up so that anything that changes
    # to predictions are caught in the unit test. This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(predictions == last_version_predictions)

    expected_info = "AdaptiveRandomForest(binary_split=False, disable_weighted_vote=False,\n" \
                    "                     drift_detection_method=ADWIN(delta=0.001), grace_period=50,\n" \
                    "                     lambda_value=6, leaf_prediction='nb',\n" \
                    "                     max_byte_size=33554432, max_features=5,\n" \
                    "                     memory_estimate_period=2000000, n_estimators=3,\n" \
                    "                     nb_threshold=0, no_preprune=False, nominal_attributes=None,\n" \
                    "                     performance_metric='acc', random_state=112,\n" \
                    "                     remove_poor_atts=False, split_confidence=0.01,\n" \
                    "                     split_criterion='info_gain', stop_mem_management=False,\n" \
                    "                     tie_threshold=0.05,\n" \
                    "                     warning_detection_method=ADWIN(delta=0.01))"
    assert learner.get_info() == expected_info


def test_adaptive_random_forests_nba():
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112, n_classes=2)
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
    last_version_predictions = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                                1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                                1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                                0, 0, 0, 1, 1, 1, 0, 0, 0]

    # Performance below does not need to be guaranteed. This check is set up so that anything that changes
    # to predictions are caught in the unit test. This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(predictions == last_version_predictions)

    expected_info = "AdaptiveRandomForest(binary_split=False, disable_weighted_vote=False,\n" \
                    "                     drift_detection_method=ADWIN(delta=0.001), grace_period=50,\n" \
                    "                     lambda_value=6, leaf_prediction='nba',\n" \
                    "                     max_byte_size=33554432, max_features=5,\n" \
                    "                     memory_estimate_period=2000000, n_estimators=3,\n" \
                    "                     nb_threshold=0, no_preprune=False, nominal_attributes=None,\n" \
                    "                     performance_metric='acc', random_state=112,\n" \
                    "                     remove_poor_atts=False, split_confidence=0.01,\n" \
                    "                     split_criterion='info_gain', stop_mem_management=False,\n" \
                    "                     tie_threshold=0.05,\n" \
                    "                     warning_detection_method=ADWIN(delta=0.01))"
    assert learner.get_info() == expected_info


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
    last_version_predictions = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]

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
    assert all_predictions.shape == (4 * 5, 2)

    last_version_predictions = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    assert type(learner.predict_proba(X)) == np.ndarray
    assert np.alltrue(all_predictions.argmax(axis=1) == last_version_predictions)
