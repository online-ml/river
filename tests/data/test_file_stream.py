import os
import numpy as np
from skmultiflow.data.file_stream import FileStream
from skmultiflow.options.file_option import FileOption


def test_random_rbf_generator(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/datasets/sea_stream.csv')
    file_option = FileOption('FILE', 'sea', test_file, 'csv', False)
    stream = FileStream(file_option)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == 40000

    expected_header = ['attrib1', 'attrib2', 'attrib3']
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 3

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 3

    assert stream.get_num_targets() == 1

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'sea_stream.csv - 2 class labels'

    assert stream.has_more_instances() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_instance()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    X, y = stream.get_last_instance()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_instance(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)
