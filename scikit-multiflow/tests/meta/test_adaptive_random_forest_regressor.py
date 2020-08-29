import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.meta import AdaptiveRandomForestRegressor


def test_adaptive_random_forest_regressor_mean():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner1 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='mean', aggregation_method='mean',
        weighted_vote_strategy=None, drift_detection_criteria='mse', max_byte_size=float('Inf'),
        random_state=1
    )
    learner2 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features=0.2, leaf_prediction='mean', aggregation_method='median',
        weighted_vote_strategy=None, drift_detection_criteria='mae', max_byte_size=float('Inf'),
        random_state=1
    )
    learner3 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='mean', aggregation_method='mean',
        weighted_vote_strategy='mse', drift_detection_criteria='predictions',
        max_byte_size=float('Inf'), random_state=1
    )

    cnt = 0
    max_samples = 500
    y_pred1 = array('d')
    y_pred2 = array('d')
    y_pred3 = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred1.append(learner1.predict(X)[0])
            y_pred2.append(learner2.predict(X)[0])
            y_pred3.append(learner3.predict(X)[0])
            y_true.append(y[0])
        learner1.partial_fit(X, y)
        learner2.partial_fit(X, y)
        learner3.partial_fit(X, y)
        cnt += 1

    error1 = mean_absolute_error(y_true, y_pred1)
    error2 = mean_absolute_error(y_true, y_pred2)
    error3 = mean_absolute_error(y_true, y_pred3)

    expected_error1 = 148.62
    expected_error2 = 144.55
    expected_error3 = 147.93

    assert np.isclose(round(error1, 2), expected_error1)
    assert np.isclose(round(error2, 2), expected_error2)
    assert np.isclose(round(error3, 2), expected_error3)

    expected_info = "AdaptiveRandomForestRegressor(aggregation_method='mean', " \
                    "binary_split=False, drift_detection_criteria='mse', " \
                    "drift_detection_method=ADWIN ( delta=0.001 ), grace_period=50, " \
                    "lambda_value=6, leaf_prediction='mean', learning_ratio_const=True, " \
                    "learning_ratio_decay=0.001, learning_ratio_perceptron=0.1, " \
                    "max_byte_size=inf, max_features=4, memory_estimate_period=2000000, " \
                    "n_estimators=3, no_preprune=False, nominal_attributes=None, " \
                    "random_state=1, remove_poor_atts=False, split_confidence=0.01, " \
                    "stop_mem_management=False, tie_threshold=0.05, " \
                    "warning_detection_method=ADWIN ( delta=0.01 ), weighted_vote_strategy=None)"
    info = " ".join([line.strip() for line in learner1.get_info().split()])
    assert info == expected_info
    assert type(learner1.predict(X)) == np.ndarray


def test_adaptive_random_forest_regressor_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner1 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='log2', leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy=None, max_byte_size=float('Inf'),
        random_state=1
    )
    learner2 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='perceptron',
        aggregation_method='median', weighted_vote_strategy=None, max_byte_size=float('Inf'),
        random_state=1
    )
    learner3 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features=4, leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy=None,  learning_ratio_const=False,
        max_byte_size=float('Inf'), random_state=1
    )

    cnt = 0
    max_samples = 500
    y_pred1 = array('d')
    y_pred2 = array('d')
    y_pred3 = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred1.append(learner1.predict(X)[0])
            y_pred2.append(learner2.predict(X)[0])
            y_pred3.append(learner3.predict(X)[0])
            y_true.append(y[0])
        learner1.partial_fit(X, y)
        learner2.partial_fit(X, y)
        learner3.partial_fit(X, y)
        cnt += 1

    error1 = mean_absolute_error(y_true, y_pred1)
    error2 = mean_absolute_error(y_true, y_pred2)
    error3 = mean_absolute_error(y_true, y_pred3)

    expected_error1 = 126.01
    expected_error2 = 126.32
    expected_error3 = 124.79

    assert np.isclose(round(error1, 2), expected_error1)
    assert np.isclose(round(error2, 2), expected_error2)
    assert np.isclose(round(error3, 2), expected_error3)

    learner1.reset()

    expected_info = "AdaptiveRandomForestRegressor(aggregation_method='median', " \
                    "binary_split=False, drift_detection_criteria='mse', " \
                    "drift_detection_method=ADWIN ( delta=0.001 ), grace_period=50, " \
                    "lambda_value=6, leaf_prediction='perceptron', learning_ratio_const=True, " \
                    "learning_ratio_decay=0.001, learning_ratio_perceptron=0.1, " \
                    "max_byte_size=inf, max_features=4, memory_estimate_period=2000000, " \
                    "n_estimators=3, no_preprune=False, nominal_attributes=None, " \
                    "random_state=1, remove_poor_atts=False, split_confidence=0.01, " \
                    "stop_mem_management=False, tie_threshold=0.05, " \
                    "warning_detection_method=ADWIN ( delta=0.01 ), weighted_vote_strategy=None)"

    info = " ".join([line.strip() for line in learner2.get_info().split()])
    assert info == expected_info


def test_adaptive_random_forest_regressor_drift_detection_coverage():
    max_samples = 1000
    random_state = np.random.RandomState(7)
    X = random_state.uniform(size=(max_samples, 10))
    threshold = np.mean(np.sum(X, axis=1))

    # ARFReg with background learner enabled
    learner1 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy=None, drift_detection_criteria='mse',
        max_byte_size=float('Inf'), random_state=1
    )
    # ARFReg without background learner
    learner2 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy=None, warning_detection_method=None,
        drift_detection_criteria='mse', max_byte_size=float('Inf'), random_state=1
    )

    cnt = 0
    y_pred1 = array('d')
    y_pred2 = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        x = X[cnt].reshape(1, -1)
        if cnt < 250:
            if np.sum(x) > threshold:
                y = np.asarray([random_state.normal(loc=5, scale=1.0)])
            else:
                y = np.asarray([random_state.normal(loc=-5, scale=1.0)])
        elif cnt < 500:  # First abrupt drift
            if np.sum(x) > threshold:
                y = np.asarray([random_state.normal(loc=10, scale=1.0)])
            else:
                y = np.asarray([random_state.normal(loc=-10, scale=1.0)])
        else:  # Second abrupt drift
            if np.sum(x) > threshold:
                y = np.asarray([random_state.normal(loc=20, scale=2.0)])
            else:
                y = np.asarray([random_state.normal(loc=-20, scale=2.0)])

        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred1.append(learner1.predict(x)[0])
            y_pred2.append(learner2.predict(x)[0])
            y_true.append(y)
        learner1.partial_fit(x, y)
        learner2.partial_fit(x, y)

        cnt += 1

    expected_error1 = 10.94
    expected_error2 = 12.08

    error1 = mean_absolute_error(y_true, y_pred1)
    error2 = mean_absolute_error(y_true, y_pred2)

    assert np.isclose(round(error1, 2), expected_error1)
    assert np.isclose(round(error2, 2), expected_error2)
