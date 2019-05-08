import os
import numpy as np
from skmultiflow.data.waveform_generator import WaveformGenerator


def test_waveform_generator(test_path):
    stream = WaveformGenerator(random_state=23, has_noise=False)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9',
                       'att_num_10', 'att_num_11', 'att_num_12', 'att_num_13', 'att_num_14',
                       'att_num_15', 'att_num_16', 'att_num_17', 'att_num_18', 'att_num_19',
                       'att_num_20']
    assert stream.feature_names == expected_names

    expected_targets = [0, 1, 2]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['target_0']

    assert stream.n_features == 21

    assert stream.n_cat_features == 0

    assert stream.n_num_features == 21

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Waveform Generator - 1 target(s), 3 classes, 21 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_stream.npz')
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


def test_waveform_generator_noise(test_path):
    # Noise test
    stream = WaveformGenerator(random_state=23, has_noise=True)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_num_5', 'att_num_6', 'att_num_7', 'att_num_8', 'att_num_9',
                       'att_num_10', 'att_num_11', 'att_num_12', 'att_num_13', 'att_num_14',
                       'att_num_15', 'att_num_16', 'att_num_17', 'att_num_18', 'att_num_19',
                       'att_num_20', 'att_num_21', 'att_num_22', 'att_num_23', 'att_num_24',
                       'att_num_25', 'att_num_26', 'att_num_27', 'att_num_28', 'att_num_29',
                       'att_num_30', 'att_num_31', 'att_num_32', 'att_num_33', 'att_num_34',
                       'att_num_35', 'att_num_36', 'att_num_37', 'att_num_38', 'att_num_39',
                       ]
    assert stream.feature_names == expected_names

    expected_targets = [0, 1, 2]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['target_0']

    assert stream.n_features == 40

    assert stream.n_cat_features == 0

    assert stream.n_num_features == 40

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Waveform Generator - 1 target(s), 3 classes, 40 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_noise_stream.npz')
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

    expected_info = "WaveformGenerator(has_noise=True, random_state=23)"
    assert stream.get_info() == expected_info
