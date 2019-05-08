import os
import numpy as np
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator


def test_random_rbf_generator(test_path):
    stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=4, n_features=10, n_centroids=50)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9']
    assert stream.feature_names == expected_names

    expected_target_values = [0, 1, 2, 3]
    assert stream.target_values == expected_target_values

    assert stream.target_names == ['target_0']

    assert stream.n_features == 10

    assert stream.n_cat_features == 0

    assert stream.n_num_features == 10

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Random RBF Generator - 1 target(s), 4 classes, 10 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'random_rbf_stream.npz')
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

    expected_info = "RandomRBFGenerator(model_random_state=99, n_centroids=50, n_classes=4,\n" \
                    "                   n_features=10, sample_random_state=50)"
    assert stream.get_info() == expected_info
