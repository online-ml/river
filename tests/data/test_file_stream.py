import os
import numpy as np
from skmultiflow.data.file_stream import FileStream


def test_file_stream(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_stream.csv')
    stream = FileStream(test_file)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == 40000

    expected_names = ['attrib1', 'attrib2', 'attrib3']
    assert stream.feature_names == expected_names

    expected_targets = [0, 1]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['class']

    assert stream.n_features == 3

    assert stream.n_cat_features == 0

    assert stream.n_num_features == 3

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'sea_stream.csv - 1 target(s), 2 classes'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
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

    expected_info = "FileStream(filename='sea_stream.csv', target_idx=-1, n_targets=1, cat_features=None)"
    assert stream.get_info() == expected_info
