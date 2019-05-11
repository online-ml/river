import os
import numpy as np
from skmultiflow.data.agrawal_generator import AGRAWALGenerator


def test_agrawal_generator(test_path):
    stream = AGRAWALGenerator(classification_function=2, random_state=112, balance_classes=False, perturbation=0.28)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
    assert stream.feature_names == expected_names

    expected_targets = [0, 1]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['target']

    assert stream.n_features == 9

    assert stream.n_cat_features == 3

    assert  stream.n_num_features == 6

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'AGRAWAL Generator - 1 target(s), 2 classes, 9 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'agrawal_stream.npz')
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
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.n_targets == np.array(y).ndim

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "AGRAWALGenerator(balance_classes=False, classification_function=2,\n" \
                    "                 perturbation=0.28, random_state=112)"
    assert stream.get_info() == expected_info


def test_agrawal_generator_all_functions(test_path):
    for f in range(10):
        stream = AGRAWALGenerator(classification_function=f, random_state=1)
        stream.prepare_for_use()

        # Load test data corresponding to first 10 instances
        test_file = os.path.join(test_path, 'agrawal_stream_{}.npz'.format(f))
        data = np.load(test_file)
        X_expected = data['X']
        y_expected = data['y']

        X, y = stream.next_sample(10)
        assert np.alltrue(X == X_expected)
        assert np.alltrue(y == y_expected)


def test_agrawal_drift(test_path):
    stream = AGRAWALGenerator(random_state=1)
    stream.prepare_for_use()
    X, y = stream.next_sample(10)
    stream.generate_drift()
    X_drift, y_drift = stream.next_sample(10)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'agrawal_stream_drift.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X = np.concatenate((X, X_drift))
    y = np.concatenate((y, y_drift))
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)
