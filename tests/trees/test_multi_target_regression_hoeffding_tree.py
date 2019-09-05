import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import MultiTargetRegressionHoeffdingTree


def test_multi_target_regression_hoeffding_tree_mean(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='mean')

    cnt = 0
    max_samples = 2000
    wait_samples = 200
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

    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 191.2823924547882
    assert np.isclose(error, expected_error)

    expected_info = "MultiTargetRegressionHoeffdingTree(binary_split=False, grace_period=200,\n" \
                    "                                   leaf_prediction='mean',\n" \
                    "                                   learning_ratio_const=True,\n" \
                    "                                   learning_ratio_decay=0.001,\n" \
                    "                                   learning_ratio_perceptron=0.02,\n" \
                    "                                   max_byte_size=33554432,\n" \
                    "                                   memory_estimate_period=1000000,\n" \
                    "                                   nb_threshold=0, no_preprune=False,\n" \
                    "                                   nominal_attributes=None, random_state=None,\n" \
                    "                                   remove_poor_atts=False,\n" \
                    "                                   split_confidence=1e-07,\n" \
                    "                                   stop_mem_management=False,\n" \
                    "                                   tie_threshold=0.05)"
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_multi_target_regression_hoeffding_tree_perceptron(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='perceptron',
                                                 random_state=1)

    cnt = 0
    max_samples = 2000
    wait_samples = 200
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
    expected_error = 144.44877909957646
    assert np.isclose(error, expected_error)

    expected_info = "MultiTargetRegressionHoeffdingTree(binary_split=False, grace_period=200,\n" \
                    "                                   leaf_prediction='perceptron',\n" \
                    "                                   learning_ratio_const=True,\n" \
                    "                                   learning_ratio_decay=0.001,\n" \
                    "                                   learning_ratio_perceptron=0.02,\n" \
                    "                                   max_byte_size=33554432,\n" \
                    "                                   memory_estimate_period=1000000,\n" \
                    "                                   nb_threshold=0, no_preprune=False,\n" \
                    "                                   nominal_attributes=None, random_state=1,\n" \
                    "                                   remove_poor_atts=False,\n" \
                    "                                   split_confidence=1e-07,\n" \
                    "                                   stop_mem_management=False,\n" \
                    "                                   tie_threshold=0.05)"
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_multi_target_regression_hoeffding_tree_adaptive(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)
    stream.prepare_for_use()

    learner = MultiTargetRegressionHoeffdingTree(leaf_prediction='adaptive',
                                                 random_state=1)

    cnt = 0
    max_samples = 2000
    wait_samples = 200
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
    expected_error = 150.63043466390528
    assert np.isclose(error, expected_error)

    expected_info = "MultiTargetRegressionHoeffdingTree(binary_split=False, grace_period=200,\n" \
                    "                                   leaf_prediction='adaptive',\n" \
                    "                                   learning_ratio_const=True,\n" \
                    "                                   learning_ratio_decay=0.001,\n" \
                    "                                   learning_ratio_perceptron=0.02,\n" \
                    "                                   max_byte_size=33554432,\n" \
                    "                                   memory_estimate_period=1000000,\n" \
                    "                                   nb_threshold=0, no_preprune=False,\n" \
                    "                                   nominal_attributes=None, random_state=1,\n" \
                    "                                   remove_poor_atts=False,\n" \
                    "                                   split_confidence=1e-07,\n" \
                    "                                   stop_mem_management=False,\n" \
                    "                                   tie_threshold=0.05)"

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
