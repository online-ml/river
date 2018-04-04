import numpy as np
from skmultiflow.data.generators.random_rbf_generator import RandomRBFGenerator


def test_random_rbf_generator():
    stream = RandomRBFGenerator(model_seed=99, instance_seed=50, num_classes=4, num_att=10, num_centroids=50)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9']
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1, 2, 3]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 10

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 10

    assert stream.get_num_targets() == 4

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'Random RBF Generator - 4 class labels'

    assert stream.has_more_instances() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    data = np.load('random_rbf_stream.npz')
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
