import os
import numpy as np
from skmultiflow.data.multilabel_generator import MultilabelGenerator


def test_multilabel_generator(test_path):
    stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == 100

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9',
                       'att_num_10', 'att_num_11', 'att_num_12', 'att_num_13', 'att_num_14',
                       'att_num_15', 'att_num_16', 'att_num_17', 'att_num_18', 'att_num_19']

    assert stream.feature_names == expected_names

    expected_targets = [[0, 1], [0, 1], [0, 1], [0, 1]]
    assert stream.target_values == expected_targets

    expected_names = ['target_0', 'target_1', 'target_2', 'target_3']

    assert stream.target_names == expected_names

    assert stream.n_features == 20

    assert stream.n_cat_features == 0

    assert stream.n_num_features == stream.n_features

    assert stream.n_targets == 4

    assert stream.get_data_info() == 'Multilabel Generator - 4 target(s), 2 classes, 20 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'multilabel_stream.npz')
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

    assert stream.n_targets == y.shape[1]

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "MultilabelGenerator(n_features=20, n_labels=4, n_samples=100, n_targets=4,\n" \
                    "                    random_state=0)"
    assert stream.get_info() == expected_info
