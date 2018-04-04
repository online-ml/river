import numpy as np
from skmultiflow.data.generators.sea_generator import SEAGenerator


def test_sea_generator():
    stream = SEAGenerator(classification_function=2, instance_seed=112, balance_classes=False, noise_percentage=0.28)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2']
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 3

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 3

    assert stream.get_num_targets() == 2

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'SEA Generator - 2 class labels'

    assert stream.has_more_instances() is True

    assert stream.has_more_mini_batch() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    data = np.load('sea_stream.npz')
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
