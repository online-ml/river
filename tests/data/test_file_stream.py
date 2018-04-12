import os
import numpy as np
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption


def test_file_stream(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/datasets/sea_stream.csv')
    file_option = FileOption('FILE', 'sea', test_file, 'csv', False)
    stream = FileStream(file_option)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == 40000

    expected_names = ['attrib1', 'attrib2', 'attrib3']
    assert stream.get_feature_names() == expected_names

    expected_targets = [0, 1]
    assert stream.get_targets() == expected_targets

    assert stream.get_target_names() == ['class']

    assert stream.get_n_features() == 3

    assert stream.get_n_cat_features() == 0

    assert stream.get_n_num_features() == 3

    assert stream.get_n_targets() == 1

    assert stream.get_name() == 'sea_stream.csv - 1 target(s), 2 classes'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream.npz')
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

    assert stream.get_n_targets() == np.array(y).ndim

    assert stream.get_n_features() == X.shape[1]
