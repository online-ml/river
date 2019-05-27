import os
import numpy as np
from skmultiflow.data.led_generator import LEDGenerator


def test_led_generator(test_path):
    stream = LEDGenerator(random_state=112, noise_percentage=0.28, has_noise=True)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2',
                      'att_num_3', 'att_num_4', 'att_num_5',
                      'att_num_6', 'att_num_7', 'att_num_8',
                      'att_num_9', 'att_num_10', 'att_num_11',
                      'att_num_12', 'att_num_13', 'att_num_14',
                      'att_num_15', 'att_num_16', 'att_num_17',
                      'att_num_18', 'att_num_19', 'att_num_20',
                      'att_num_21', 'att_num_22', 'att_num_23']

    assert stream.feature_names == expected_names

    expected_targets = [i for i in range(10)]
    assert stream.target_values == expected_targets

    assert stream.target_names is None

    assert stream.n_features == 24

    assert stream.n_cat_features == stream.n_features

    assert stream.n_num_features == 0

    assert stream.n_targets == 1

    assert stream.n_classes == 10

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'led_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "LEDGenerator(has_noise=True, noise_percentage=0.28, random_state=112)"
    assert stream.get_info() == expected_info
