import os
import numpy as np
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator


def test_random_tree_generator(test_path):
    stream = RandomTreeGenerator(tree_seed=23, instance_seed=12, n_classes=2, n_nominal_attributes=2,
                                 n_numerical_attributes=5, n_values_per_nominal=5, max_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_nom_0_val0', 'att_nom_0_val1', 'att_nom_0_val2', 'att_nom_0_val3', 'att_nom_0_val4',
                       'att_nom_1_val0', 'att_nom_1_val1', 'att_nom_1_val2', 'att_nom_1_val3', 'att_nom_1_val4']
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 15

    assert stream.get_num_nominal_attributes() == 2

    assert stream.get_num_numerical_attributes() == 5

    assert stream.get_num_classes() == 2

    assert stream.get_num_values_per_nominal_attribute() == 5

    assert stream.get_plot_name() == 'Random Tree Generator - 2 class labels'

    assert stream.has_more_instances() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'random_tree_stream.npz')
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
