import os
import numpy as np
import pytest

from skmultiflow.data import ConceptDriftStream


def test_concept_drift_stream(test_path):
    stream = ConceptDriftStream(random_state=1, position=20, width=5)

    assert stream.n_remaining_samples() == -1

    expected_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue",
                      "hyears", "loan"]
    assert stream.feature_names == expected_names

    expected_targets = [0, 1]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['target']

    assert stream.n_features == 9

    assert stream.n_cat_features == 3

    assert stream.n_num_features == 6

    assert stream.n_targets == 1

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'concept_drift_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    X, y = stream.last_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_sample(30)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.n_targets == np.array(y).ndim

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "ConceptDriftStream(alpha=None,\n" \
                    "                   drift_stream=AGRAWALGenerator(balance_classes=False,\n" \
                    "                                                 classification_function=2,\n" \
                    "                                                 perturbation=0.0,\n" \
                    "                                                 random_state=112),\n" \
                    "                   position=20, random_state=1,\n" \
                    "                   stream=AGRAWALGenerator(balance_classes=False,\n" \
                    "                                           classification_function=0,\n" \
                    "                                           perturbation=0.0, random_state=112),\n" \
                    "                   width=5)"
    assert stream.get_info() == expected_info


def test_concept_drift_stream_with_alpha(test_path):
    stream = ConceptDriftStream(alpha=0.01, random_state=1, position=20)

    expected_info = "ConceptDriftStream(alpha=0.01,\n" \
                    "                   drift_stream=AGRAWALGenerator(balance_classes=False,\n" \
                    "                                                 classification_function=2,\n" \
                    "                                                 perturbation=0.0,\n" \
                    "                                                 random_state=112),\n" \
                    "                   position=20, random_state=1,\n" \
                    "                   stream=AGRAWALGenerator(balance_classes=False,\n" \
                    "                                           classification_function=0,\n" \
                    "                                           perturbation=0.0, random_state=112),\n" \
                    "                   width=5729)"
    assert stream.get_info() == expected_info

    with pytest.warns(FutureWarning) as actual_warning:
        ConceptDriftStream(alpha=0, random_state=1, position=20)

    assert actual_warning[0].message.args[0] == "Default value for 'alpha' has changed from 0 " \
                                            "to None. 'alpha=0' will throw an error from v0.7.0"

