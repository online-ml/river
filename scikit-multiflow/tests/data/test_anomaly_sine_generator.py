import os

import pytest

import numpy as np

from skmultiflow.data import AnomalySineGenerator


def test_anomaly_sine_generator(test_path):
    stream = AnomalySineGenerator(random_state=12345,
                                  n_samples=100,
                                  n_anomalies=25,
                                  contextual=True,
                                  n_contextual=10)

    assert stream.n_remaining_samples() == 100

    expected_names = ['att_idx_0', 'att_idx_1']
    assert stream.feature_names == expected_names

    expected_targets = [0, 1]
    assert stream.target_values == expected_targets

    assert stream.target_names == ['anomaly']

    assert stream.n_features == 2

    assert stream.n_cat_features == 0

    assert stream.n_num_features == 2

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Anomaly Sine Generator - 1 target(s), 2 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'anomaly_sine_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(np.isclose(X[0], X_expected[0]))
    assert np.alltrue(np.isclose(y[0], y_expected[0]))

    X, y = stream.last_sample()
    assert np.alltrue(np.isclose(X[0], X_expected[0]))
    assert np.alltrue(np.isclose(y[0], y_expected[0]))

    stream.restart()
    X, y = stream.next_sample(100)
    assert np.alltrue(np.isclose(X, X_expected))
    assert np.alltrue(np.isclose(y, y_expected))

    assert stream.n_targets == np.array(y).ndim

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "AnomalySineGenerator(contextual=True, n_anomalies=25, n_contextual=10, " \
                    "n_samples=100, noise=0.5, random_state=12345, replace=True, shift=4)"
    info = " ".join([line.strip() for line in stream.get_info().split()])
    assert info == expected_info

    # Coverage
    with pytest.raises(ValueError):
        # Invalid n_anomalies
        AnomalySineGenerator(n_samples=100, n_anomalies=250)

    with pytest.raises(ValueError):
        # Invalid n_contextual
        AnomalySineGenerator(n_samples=100, n_anomalies=50, contextual=True, n_contextual=250)
