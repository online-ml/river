import os
import numpy as np
from skmultiflow.data.generators.sea_generator import SEAGenerator


def test_sea_generator(test_path):
    stream = SEAGenerator(classification_function=2, sample_seed=112, balance_classes=False, noise_percentage=0.28)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_header = ['att_num_0', 'att_num_1', 'att_num_2']
    assert stream.get_features_labels() == expected_header

    expected_classes = [0, 1]
    assert stream.get_classes() == expected_classes

    assert stream.get_output_labels() == ['class']

    assert stream.get_n_features() == 3

    assert stream.get_n_cat_features() == 0

    assert stream.get_n_num_features() == 3

    assert stream.get_n_classes() == 2

    assert stream.get_plot_name() == 'SEA Generator - 2 class labels'

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

    X, y = stream.get_last_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)
