import numpy as np

from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier


def test_adaptive_random_forests_mc():
    stream = RandomTreeGenerator(tree_random_state=112,
                                 sample_random_state=112,
                                 n_classes=2)

    learner = AdaptiveRandomForestClassifier(n_estimators=3,
                                             leaf_prediction='mc',
                                             random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

    cnt = 0
    max_samples = 5000
    y_pred = []
    true_labels = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(int(learner.predict(X)[0]))
            true_labels.append(y[0])

        learner.partial_fit(X, y)
        cnt += 1
    y_pred_expected = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                       1, 1, 0, 0, 1, 0, 1, 1, 1, 1,
                       1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                       0, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                       0, 0, 0, 1, 1, 1, 0, 0, 0]

    # Performance below does not need to be guaranteed. This check is set up so
    # that anything that changes to predictions are caught in the unit test.
    # This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(y_pred == y_pred_expected)

    expected_info = "AdaptiveRandomForestClassifier(binary_split=False, " \
                    "disable_weighted_vote=False, drift_detection_method=ADWIN ( delta=0.001 ), " \
                    "grace_period=50, lambda_value=6, leaf_prediction='mc', " \
                    "max_byte_size=33554432, max_features=5, memory_estimate_period=2000000, " \
                    "n_estimators=3, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, performance_metric='acc', random_state=112, " \
                    "remove_poor_atts=False, split_confidence=0.01, " \
                    "split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05, warning_detection_method=ADWIN ( delta=0.01 ))"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_adaptive_random_forests_nb():
    stream = RandomTreeGenerator(tree_random_state=112,
                                 sample_random_state=112,
                                 n_classes=2)

    learner = AdaptiveRandomForestClassifier(n_estimators=3,
                                             random_state=112,
                                             leaf_prediction='nb')

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y)

    cnt = 0
    max_samples = 5000
    predictions = []
    true_labels = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(int(learner.predict(X)[0]))
            true_labels.append(y[0])

        learner.partial_fit(X, y)
        cnt += 1

    last_version_predictions = [1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                                1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
                                1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                                0, 1, 0, 1, 1, 1, 0, 0, 0]

    # Performance below does not need to be guaranteed. This check is set up so
    # that anything that changes  to predictions are caught in the unit test.
    # This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(predictions == last_version_predictions)

    expected_info = "AdaptiveRandomForestClassifier(binary_split=False, " \
                    "disable_weighted_vote=False, drift_detection_method=ADWIN ( delta=0.001 ), " \
                    "grace_period=50, lambda_value=6, leaf_prediction='nb', " \
                    "max_byte_size=33554432, max_features=5, memory_estimate_period=2000000, " \
                    "n_estimators=3, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, performance_metric='acc', random_state=112, " \
                    "remove_poor_atts=False, split_confidence=0.01, " \
                    "split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05, warning_detection_method=ADWIN ( delta=0.01 ))"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_adaptive_random_forests_nba():
    stream = RandomTreeGenerator(tree_random_state=112,
                                 sample_random_state=112,
                                 n_classes=2)

    learner = AdaptiveRandomForestClassifier(n_estimators=3,
                                             random_state=112)

    X, y = stream.next_sample(150)
    learner.partial_fit(X, y, classes=[0, 1])  # labels given

    cnt = 0
    max_samples = 5000
    y_proba = []
    true_labels = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_proba.append(learner.predict_proba(X)[0])
            true_labels.append(y[0])

        learner.partial_fit(X, y)
        cnt += 1

    assert np.alltrue([np.isclose(probabilities.sum(), 1) for probabilities in y_proba]), \
        "Probabilities should sum to 1."

    y_proba = np.asarray(y_proba).squeeze()
    assert y_proba.shape == (49, 2)

    y_pred = y_proba.argmax(axis=1)
    y_pred_expected = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1,
                       1, 1, 1, 0, 1, 0, 1, 1, 0, 1,
                       1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                       0, 0, 0, 1, 1, 1, 0, 0, 0]

    # Performance below does not need to be guaranteed. This check is set up so
    # that anything that changes to predictions are caught in the unit test.
    # This helps prevent accidental changes.

    assert type(learner.predict(X)) == np.ndarray
    assert np.alltrue(y_pred == y_pred_expected)

    expected_info = "AdaptiveRandomForestClassifier(binary_split=False, " \
                    "disable_weighted_vote=False, drift_detection_method=ADWIN ( delta=0.001 ), " \
                    "grace_period=50, lambda_value=6, leaf_prediction='nba', " \
                    "max_byte_size=33554432, max_features=5, memory_estimate_period=2000000, " \
                    "n_estimators=3, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, performance_metric='acc', random_state=112, " \
                    "remove_poor_atts=False, split_confidence=0.01, " \
                    "split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05, warning_detection_method=ADWIN ( delta=0.01 ))"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info
