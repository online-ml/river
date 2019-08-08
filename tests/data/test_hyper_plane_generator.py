import os
import numpy as np
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator


def test_hyper_plane_generator(test_path):

    stream = HyperplaneGenerator(random_state=112, n_features=10, n_drift_features=2, mag_change=0.6,
                                 noise_percentage=0.28, sigma_percentage=0.1)
    stream.prepare_for_use()

    n_features = 10
    assert stream.n_remaining_samples() == -1

    expected_names = []
    for i in range(n_features):
        expected_names.append("att_num_" + str(i))
    assert stream.feature_names == expected_names

    assert stream.target_values == [0, 1]

    assert stream.target_names == ["target_0"]

    assert stream.n_features == n_features

    assert stream.n_cat_features == 0

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Hyperplane Generator - 1 target(s), 2 classes, 10 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'hyper_plane_stream.npz')
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

    expected_info = "HyperplaneGenerator(mag_change=0.6, n_drift_features=2, n_features=10,\n" \
                    "                    noise_percentage=0.28, random_state=112,\n" \
                    "                    sigma_percentage=0.1)"
    assert stream.get_info() == expected_info

    ### test calculation of sum of weights, and sum of weights+features
    batch_size = 10
    n_features = 2
    stream = HyperplaneGenerator(random_state=112, n_features=n_features, n_drift_features=2, mag_change=0.6,
                                 noise_percentage=0.0, sigma_percentage=0.1)
    stream.prepare_for_use()

    # check features and weights
    X, y = stream.next_sample(batch_size)
    weights = stream._weights
    w = np.array([0.9750571288732851, 1.2403046199226442])
    data = np.array([[0.950016579405167, 0.07567720470206152],
                     [0.8327457625618593, 0.054805740282408255],
                     [0.8853514580727667, 0.7223465108072455],
                     [0.9811992777207516, 0.34341985076716164],
                     [0.39464258869483526, 0.004944924811720708],
                     [0.9558068694855607, 0.8206093713145775],
                     [0.378544457805313, 0.7847636149698817],
                     [0.5460739378008381, 0.1622260202888307],
                     [0.04500817232778065, 0.33218775732993966],
                     [0.8392114852107733, 0.7093616146129875]])

    assert np.alltrue(weights == w)
    assert np.alltrue(X == data)

    # check labels
    labels = np.zeros([1, batch_size])
    sum_weights = np.sum(weights)
    for i in range(batch_size):
        if weights[0] * data[i, 0] + weights[1] * data[i, 1] >= 0.5 * sum_weights:
            labels[0, i] = 1

    assert np.alltrue(y == labels)



