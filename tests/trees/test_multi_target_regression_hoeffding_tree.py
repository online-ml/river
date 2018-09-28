import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import MultiTargetRegressionHoeffdingTree


def test_multi_target_regression_hoeffding_tree_mean(test_path):
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='mean')

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    test_file = os.path.join(test_path,
                             'expected_preds_multi_target_regression_mean.npy')
    expected_predictions = np.load(test_file)

    # print(expected_predictions.shape)
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 167.40626294018753
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: mean - nb_threshold: 0 - ' \
        'nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_multi_target_regression_hoeffding_tree_perceptron(test_path):
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='perceptron',
                                                 random_state=1)

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    test_file = os.path.join(
        test_path,
        'expected_preds_multi_target_regression_perceptron.npy'
    )
    expected_predictions = np.load(test_file)

    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)
    expected_error = 134.5949775942447
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: perceptron - ' \
        'nb_threshold: 0 - nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_multi_target_regression_hoeffding_tree_adaptive(test_path):
    stream = RegressionGenerator(n_samples=500, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='adaptive',
                                                 random_state=1)

    cnt = 0
    max_samples = 500
    wait_samples = 10
    y_pred = np.zeros((int(max_samples / wait_samples), 3))
    y_true = np.zeros((int(max_samples / wait_samples), 3))

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred[int(cnt / wait_samples), :] = learner.predict(X)
            y_true[int(cnt / wait_samples), :] = y
        learner.partial_fit(X, y)
        cnt += 1

    test_file = os.path.join(
        test_path,
        'expected_preds_multi_target_regression_adaptive.npy'
    )
    expected_predictions = np.load(test_file)

    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)
    expected_error = 134.43981366919218
    assert np.isclose(error, expected_error)

    expected_info = \
        'MultiTargetRegressionHoeffdingTree: max_byte_size: 33554432 - ' \
        'memory_estimate_period: 1000000 - grace_period: 200 - ' \
        'split_criterion: intra cluster variance reduction - ' \
        'split_confidence: 1e-07 - tie_threshold: 0.05 - binary_split: False' \
        ' - stop_mem_management: False - remove_poor_atts: False ' \
        '- no_pre_prune: False - leaf_prediction: adaptive - ' \
        'nb_threshold: 0 - nominal_attributes: [] - '
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_hoeffding_tree_coverage(test_path):
    # Cover nominal attribute observer
    test_file = os.path.join(test_path, 'multi_target_regression_data.npz')
    data = np.load(test_file)
    X = data['X']
    Y = data['Y']

    learner = MultiTargetRegressionHoeffdingTree(
                leaf_prediction='mean',
                nominal_attributes=[i for i in range(3)]
              )
    learner.partial_fit(X, Y)


if __name__ == '__main__':
    test_multi_target_regression_hoeffding_tree_adaptive('.')
