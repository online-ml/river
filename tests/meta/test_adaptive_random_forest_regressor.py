import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.meta import AdaptiveRandomForestRegressor


def test_adaptive_random_forest_regressor_mean():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner1 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='mean', aggregation_method='mean',
        weighted_vote_strategy=None, drift_detection_criteria='mse', random_state=1
    )
    learner2 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features=0.2, leaf_prediction='mean', aggregation_method='median',
        weighted_vote_strategy=None, drift_detection_criteria='mae', random_state=1
    )
    learner3 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='mean', aggregation_method='mean',
        weighted_vote_strategy='mse', drift_detection_criteria='predictions', random_state=1
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

    expected_error1 = 137.96879312181113
    expected_error2 = 141.07333987006018
    expected_error3 = 138.0064069056026
    assert np.isclose(error1, expected_error1)
    assert np.isclose(error2, expected_error2)
    assert np.isclose(error3, expected_error3)

    expected_info = "AdaptiveRandomForestRegressor(aggregation_method='mean', " \
                    "binary_split=False, drift_detection_criteria='mse', " \
                    "drift_detection_method=ADWIN(delta=0.001), grace_period=200, " \
                    "lambda_value=6, leaf_prediction='mean', learning_ratio_const=True, " \
                    "learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, max_features=4, memory_estimate_period=2000000, " \
                    "n_estimators=3, no_preprune=False, nominal_attributes=None, " \
                    "random_state=1, remove_poor_atts=False, split_confidence=0.01, " \
                    "stop_mem_management=False, tie_threshold=0.05, " \
                    "warning_detection_method=ADWIN(delta=0.01), weighted_vote_strategy=None)"
    info = " ".join([line.strip() for line in learner1.get_info().split()])
    assert info == expected_info
    assert type(learner1.predict(X)) == np.ndarray


def test_adaptive_random_forest_regressor_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)

    learner1 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='log2', leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy='mae', random_state=1
    )
    learner2 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features='auto', leaf_prediction='perceptron',
        aggregation_method='median', weighted_vote_strategy=None, random_state=1
    )
    learner3 = AdaptiveRandomForestRegressor(
        n_estimators=3, max_features=4, leaf_prediction='perceptron',
        aggregation_method='mean', weighted_vote_strategy='mse', random_state=1,
        learning_ratio_const=False
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

    expected_error1 = 112.70239337636369
    expected_error2 = 111.75976199439144
    expected_error3 = 113.09546750256585
    assert np.isclose(error1, expected_error1)
    assert np.isclose(error2, expected_error2)
    assert np.isclose(error3, expected_error3)

    learner1.reset()

    expected_info = "AdaptiveRandomForestRegressor(aggregation_method='median', " \
                    "binary_split=False, drift_detection_criteria='mse', " \
                    "drift_detection_method=ADWIN(delta=0.001), grace_period=200, " \
                    "lambda_value=6, leaf_prediction='perceptron', learning_ratio_const=True, " \
                    "learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, max_features=4, memory_estimate_period=2000000, " \
                    "n_estimators=3, no_preprune=False, nominal_attributes=None, " \
                    "random_state=1, remove_poor_atts=False, split_confidence=0.01, " \
                    "stop_mem_management=False, tie_threshold=0.05, " \
                    "warning_detection_method=ADWIN(delta=0.01), weighted_vote_strategy=None)"

    info = " ".join([line.strip() for line in learner2.get_info().split()])
    assert info == expected_info
