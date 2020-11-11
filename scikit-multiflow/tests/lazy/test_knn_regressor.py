from array import array

from skmultiflow.lazy import KNNRegressor
from skmultiflow.data import RegressionGenerator

from sklearn.metrics import mean_squared_error

import numpy as np


def test_knn_mean():
    n_samples = 5000
    n_wait = 100

    stream = RegressionGenerator(random_state=1,
                                 n_samples=n_samples,
                                 n_features=10)
    learner = KNNRegressor(n_neighbors=3,
                           max_window_size=1000,
                           leaf_size=40)

    expected_mse = 11538.991159236244
    run_prequential_supervised(stream=stream,
                               learner=learner,
                               max_samples=n_samples,
                               n_wait=n_wait,
                               mse=expected_mse)

    expected_info = "KNNRegressor(aggregation_method='mean', leaf_size=40, " \
                    "max_window_size=1000, metric='euclidean', n_neighbors=3)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    learner.reset()
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_knn_median():
    n_samples = 5000
    n_wait = 100

    stream = RegressionGenerator(random_state=1,
                                 n_samples=5000,
                                 n_features=10)
    learner = KNNRegressor(n_neighbors=3,
                           max_window_size=1000,
                           leaf_size=40,
                           aggregation_method='median')

    expected_mse = 14367.276807158994
    run_prequential_supervised(stream=stream,
                               learner=learner,
                               max_samples=n_samples,
                               n_wait=n_wait,
                               mse=expected_mse)


def run_prequential_supervised(stream, learner, max_samples, n_wait, mse):
    cnt = 0
    y_true = array('d')
    y_pred = array('d')

    while cnt < max_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        # Test every n samples
        if cnt % n_wait == 0:
            y_true.append(y[0])
            y_pred.append(learner.predict(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    assert np.isclose(mean_squared_error(y_true, y_pred), mse)
