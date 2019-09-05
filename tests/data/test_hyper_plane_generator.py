import os
import numpy as np
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator


def test_hyper_plane_generator(test_path):
    n_features = 10
    n_samples = 10
    stream = HyperplaneGenerator(random_state=112, n_features=n_features, n_drift_features=2, mag_change=0.6,
                                 noise_percentage=0.28, sigma_percentage=0.1)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = []
    for i in range(n_features):
        expected_names.append("att_num_" + str(i))
    assert stream.feature_names == expected_names

    assert stream.target_values == [0, 1]

    assert stream.target_names == ["target_0"]

    assert stream.n_features == n_features

    assert stream.n_cat_features == 0

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Hyperplane Generator - 1 target(s), 2 classes, 10 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'hyper_plane_stream.npz')
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
    X = np.zeros(shape=(n_samples, n_features))
    y = np.zeros(shape=(n_samples,)).astype('int')
    for i in range(n_samples):
        X[i, :], y[i] = stream.next_sample()
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    stream.restart()
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.n_targets == np.array(y).ndim

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "HyperplaneGenerator(mag_change=0.6, n_drift_features=2, n_features=10,\n" \
                    "                    noise_percentage=0.28, random_state=112,\n" \
                    "                    sigma_percentage=0.1)"
    assert stream.get_info() == expected_info
