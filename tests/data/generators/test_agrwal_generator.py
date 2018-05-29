import os
import numpy as np
from skmultiflow.data.generators.agrawal_generator import AGRAWALGenerator


def test_agrawal_generator(test_path):
    stream = AGRAWALGenerator(classification_function=2, random_state=112, balance_classes=False, perturbation=0.28)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ["salary", "commission", "age", "elevel", "car", "zipcode", "hvalue", "hyears", "loan"]
    assert stream.feature_names == expected_names

    expected_targets = [0, 1]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['target_0']

    assert stream.n_features == 9

    assert stream.n_cat_features == 3

    assert  stream.n_num_features == 6

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'AGRAWAL Generator - 1 targets, 2 classes, 9 features'

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
