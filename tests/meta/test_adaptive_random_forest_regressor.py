import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.meta import AdaptiveRandomForestRegressor


def test_adaptive_random_forest_regressor():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)
    stream.prepare_for_use()

    learner = AdaptiveRandomForestRegressor(random_state=1)

    cnt = 0
    max_samples = 500
    y_pred = array('d')
    y_true = array('d')
    y_baseline = array('d')
    wait_samples = 10

    for cnt in range(max_samples):
        X, y = stream.next_sample()
        # Test every 'wait_samples' samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_true.append(y[0])
        learner.partial_fit(X, y)

    error = mean_absolute_error(y_true, y_pred)
    assert np.isclose(error, 304, atol=2)

    assert type(learner.predict(X)) == np.ndarray

    expected_info = "AdaptiveRandomForestRegressor(binary_split=False,\n" \
                    "                              drift_detection_method=ADWIN(delta=0.001),\n" \
                    "                              grace_period=200, lambda_value=6,\n" \
                    "                              leaf_prediction='perceptron',\n" \
                    "                              learning_ratio_const=True,\n" \
                    "                              learning_ratio_decay=0.001,\n" \
                    "                              learning_ratio_perceptron=0.02,\n" \
                    "                              max_byte_size=33554432, max_features=4,\n" \
                    "                              memory_estimate_period=1000000, n_estimators=10,\n" \
                    "                              no_preprune=False, nominal_attributes=None,\n" \
                    "                              random_state=1, remove_poor_atts=False,\n" \
                    "                              split_confidence=1e-07, stop_mem_management=False,\n" \
                    "                              tie_threshold=0.05,\n" \
                    "                              warning_detection_method=ADWIN(delta=0.01))"

    assert learner.get_info() == expected_info
