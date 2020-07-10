import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
from skmultiflow.utils import calculate_object_size
from difflib import SequenceMatcher


def test_stacked_single_target_hoeffding_tree_regressor_perceptron(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)

    learner = StackedSingleTargetHoeffdingTreeRegressor(
        leaf_prediction='perceptron',
        random_state=1
    )

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
        'expected_preds_stacked_single_target_hoeffding_tree_perceptron.npy'
    )
    expected_predictions = np.load(test_file)
    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)

    expected_error = 154.59132974633712
    assert np.isclose(error, expected_error)

    expected_info = "StackedSingleTargetHoeffdingTreeRegressor(binary_split=False, grace_period=200,\n" \
                    "                                          leaf_prediction='perceptron',\n" \
                    "                                          learning_ratio_const=True,\n" \
                    "                                          learning_ratio_decay=0.001,\n" \
                    "                                          learning_ratio_perceptron=0.02,\n" \
                    "                                          max_byte_size=33554432,\n" \
                    "                                          memory_estimate_period=1000000,\n" \
                    "                                          nb_threshold=0, no_preprune=False,\n" \
                    "                                          nominal_attributes=None,\n" \
                    "                                          random_state=1,\n" \
                    "                                          remove_poor_atts=False,\n" \
                    "                                          split_confidence=1e-07,\n" \
                    "                                          stop_mem_management=False,\n" \
                    "                                          tie_threshold=0.05)"
    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_stacked_single_target_hoeffding_tree_regressor_adaptive(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)

    learner = StackedSingleTargetHoeffdingTreeRegressor(
        leaf_prediction='adaptive',
        random_state=1
    )

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
        'expected_preds_stacked_single_target_hoeffding_tree_adaptive.npy'
    )
    expected_predictions = np.load(test_file)

    assert np.allclose(y_pred, expected_predictions)
    error = mean_absolute_error(y_true, y_pred)

    expected_error = 152.8716829154756
    assert np.isclose(error, expected_error)

    expected_info = "StackedSingleTargetHoeffdingTreeRegressor(binary_split=False, grace_period=200,\n" \
                    "                                          leaf_prediction='adaptive',\n" \
                    "                                          learning_ratio_const=True,\n" \
                    "                                          learning_ratio_decay=0.001,\n" \
                    "                                          learning_ratio_perceptron=0.02,\n" \
                    "                                          max_byte_size=33554432,\n" \
                    "                                          memory_estimate_period=1000000,\n" \
                    "                                          nb_threshold=0, no_preprune=False,\n" \
                    "                                          nominal_attributes=None,\n" \
                    "                                          random_state=1,\n" \
                    "                                          remove_poor_atts=False,\n" \
                    "                                          split_confidence=1e-07,\n" \
                    "                                          stop_mem_management=False,\n" \
                    "                                          tie_threshold=0.05)"

    assert learner.get_info() == expected_info
    assert isinstance(learner.get_model_description(), type(''))


def test_stacked_single_target_hoeffding_tree_regressor_coverage():
    max_samples = 1000
    max_size_mb = 1

    stream = RegressionGenerator(
        n_samples=max_samples, n_features=10, n_informative=7, n_targets=3,
        random_state=42
    )
    X, y = stream.next_sample(max_samples)

    # Will generate a warning concerning the invalid leaf prediction option
    tree = StackedSingleTargetHoeffdingTreeRegressor(
        leaf_prediction='mean', grace_period=200,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )

    # Trying to predict without fitting
    tree.predict(X[0])

    tree.partial_fit(X, y)

    # A tree without memory management enabled reaches almost 2 MB in size
    assert calculate_object_size(tree, 'MB') <= max_size_mb

    tree = StackedSingleTargetHoeffdingTreeRegressor(
        leaf_prediction='adaptive', grace_period=200,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20,
        learning_ratio_const=False
    )
    tree.partial_fit(X, y)
    assert calculate_object_size(tree, 'MB') <= max_size_mb


def test_stacked_single_target_hoeffding_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    X, y = stream[:, :-2], stream[:, -2:]

    nominal_attr_idx = np.arange(8)
    learner = StackedSingleTargetHoeffdingTreeRegressor(
        nominal_attributes=nominal_attr_idx,
        leaf_prediction='perceptron'

    )

    learner.partial_fit(X, y)

    expected_description = "if Attribute 0 = -15.0:\n" \
                           "  if Attribute 3 = 0.0:\n" \
                           "    Leaf = Statistics {0: 80.0000, 1: [-192.4417, 80.0908], 2: [464.1268, 80.1882]}\n" \
                           "  if Attribute 3 = 1.0:\n" \
                           "    Leaf = Statistics {0: 77.0000, 1: [-184.8333, -7.2503], 2: [444.9068, 42.7423]}\n" \
                           "  if Attribute 3 = 2.0:\n" \
                           "    Leaf = Statistics {0: 56.0000, 1: [-134.1829, -1.0863], 2: [322.1336, 28.1218]}\n" \
                           "  if Attribute 3 = 3.0:\n" \
                           "    Leaf = Statistics {0: 62.0000, 1: [-148.2397, -17.2837], 2: [355.5327, 38.6913]}\n" \
                           "if Attribute 0 = 0.0:\n" \
                           "  Leaf = Statistics {0: 672.0000, 1: [390.6773, 672.0472], 2: [761.0744, 672.1686]}\n" \
                           "if Attribute 0 = 1.0:\n" \
                           "  Leaf = Statistics {0: 644.0000, 1: [671.3479, 174.3011], 2: [927.5194, 466.7064]}\n" \
                           "if Attribute 0 = 2.0:\n" \
                           "  Leaf = Statistics {0: 619.0000, 1: [867.2865, 320.6506], 2: [1262.0992, 435.2835]}\n" \
                           "if Attribute 0 = 3.0:\n" \
                           "  Leaf = Statistics {0: 627.0000, 1: [987.0864, 331.0822], 2: [1583.8108, 484.0456]}\n" \
                           "if Attribute 0 = -30.0:\n" \
                           "  Leaf = Statistics {0: 88.0000, 1: [-269.7967, 25.9328], 2: [828.2289, 57.6501]}\n"

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9

    new_sample = X[0].copy()
    # Adding a new category on the split feature
    new_sample[0] = 7
    learner.predict([new_sample])

    # Let's do the same considering other prediction strategy
    learner = StackedSingleTargetHoeffdingTreeRegressor(
        nominal_attributes=nominal_attr_idx,
        leaf_prediction='adaptive'
    )

    learner.partial_fit(X, y)
    learner.predict([new_sample])
