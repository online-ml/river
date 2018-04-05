import os
import numpy as np
from skmultiflow.data.generators.waveform_generator import WaveformGenerator


def test_waveform_generator(test_path):
    stream = WaveformGenerator(seed=23, add_noise=False)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9',
                       'att_num_10', 'att_num_11', 'att_num_12', 'att_num_13', 'att_num_14',
                       'att_num_15', 'att_num_16', 'att_num_17', 'att_num_18', 'att_num_19',
                       'att_num_20']
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1, 2]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 21

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 21

    assert stream.get_num_targets() == 3

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'Waveform Generator - 3 class labels'

    assert stream.has_more_instances() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_stream.npz')
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

    # Noise test

    stream = WaveformGenerator(seed=23, add_noise=True)
    stream.prepare_for_use()

    assert stream.estimated_remaining_instances() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9',
                       'att_num_10', 'att_num_11', 'att_num_12', 'att_num_13', 'att_num_14',
                       'att_num_15', 'att_num_16', 'att_num_17', 'att_num_18', 'att_num_19',
                       'att_num_20', 'att_num_21', 'att_num_22', 'att_num_23', 'att_num_24',
                       'att_num_25', 'att_num_26', 'att_num_27', 'att_num_28', 'att_num_29',
                       'att_num_30', 'att_num_31', 'att_num_32', 'att_num_33', 'att_num_34',
                       'att_num_35', 'att_num_36', 'att_num_37', 'att_num_38', 'att_num_39',
                       ]
    assert stream.get_attributes_header() == expected_header

    expected_classes = [0, 1, 2]
    assert stream.get_classes() == expected_classes

    assert stream.get_classes_header() == ['class']

    assert stream.get_num_attributes() == 40

    assert stream.get_num_nominal_attributes() == 0

    assert stream.get_num_numerical_attributes() == 40

    assert stream.get_num_targets() == 3

    assert stream.get_num_values_per_nominal_attribute() == 0

    assert stream.get_plot_name() == 'Waveform Generator - 3 class labels'

    assert stream.has_more_instances() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_noise_stream.npz')
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
