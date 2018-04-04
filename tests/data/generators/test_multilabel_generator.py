import numpy as np
from skmultiflow.data.generators.multilabel_generator import MultilabelGenerator

def test_multilabel_generator():
    stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == 100

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9']
    assert stream.get_attributes_header() == None

    expected_classes = [0, 1]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == None  # ['class']

    assert stream.get_num_attributes() == 20

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 20

    assert stream.get_num_targets() == 4

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'Multilabel Generator'

    assert stream.has_more_instances() is True

    assert stream.has_more_mini_batch() is None

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    data = np.load('multilabel_stream.npz')
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
