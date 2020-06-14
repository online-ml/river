import os
from difflib import SequenceMatcher

import numpy as np

from sklearn.metrics import mean_absolute_error

from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.utils import calculate_object_size

import pytest


def test_isoup_tree_mean(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)

    learner = iSOUPTreeRegressor(leaf_prediction='mean')

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

    expected_info = "iSOUPTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='mean', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=None, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert type(learner.predict(X)) == np.ndarray


def test_isoup_tree_perceptron(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)

    learner = iSOUPTreeRegressor(leaf_prediction='perceptron',
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

    expected_error = 148.36534180008894
    assert np.isclose(error, expected_error)

    expected_info = "iSOUPTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='perceptron', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_isoup_tree_adaptive(test_path):
    stream = RegressionGenerator(n_samples=2000, n_features=20,
                                 n_informative=15, random_state=1,
                                 n_targets=3)

    learner = iSOUPTreeRegressor(leaf_prediction='adaptive',
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

    expected_error = 152.8716829154756
    assert np.isclose(error, expected_error)

    expected_info = "iSOUPTreeRegressor(binary_split=False, grace_period=200, leaf_prediction='adaptive', " \
                    "learning_ratio_const=True, learning_ratio_decay=0.001, learning_ratio_perceptron=0.02, " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=None, random_state=1, remove_poor_atts=False, split_confidence=1e-07, " \
                    "stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_isoup_tree_coverage():
    max_samples = 1000
    max_size_mb = 1

    stream = RegressionGenerator(
        n_samples=max_samples, n_features=10, n_informative=7, n_targets=3,
        random_state=42
    )

    # Cover memory management
    tree = iSOUPTreeRegressor(
        leaf_prediction='mean', grace_period=200,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )
    # Invalid split_criterion
    tree.split_criterion = 'ICVR'

    X, y = stream.next_sample(max_samples)
    tree.partial_fit(X, y)

    # A tree without memory management enabled reaches almost 2 MB in size
    assert calculate_object_size(tree, 'MB') <= max_size_mb

    # Memory management in a tree with perceptron leaves (purposeful typo in leaf_prediction)
    tree = iSOUPTreeRegressor(
        leaf_prediction='PERCEPTRON', grace_period=200,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )
    tree.partial_fit(X, y)
    assert calculate_object_size(tree, 'MB') <= max_size_mb

    # Memory management in a tree with adaptive leaves
    tree = iSOUPTreeRegressor(
        leaf_prediction='adaptive', grace_period=200,
        memory_estimate_period=100, max_byte_size=max_size_mb*2**20
    )

    tree.partial_fit(X, y)
    assert calculate_object_size(tree, 'MB') <= max_size_mb


def test_isoup_tree_model_description():
    stream = RegressionGenerator(
        n_samples=700, n_features=20, n_informative=15, random_state=1,
        n_targets=3
    )

    learner = iSOUPTreeRegressor(leaf_prediction='mean')

    max_samples = 700
    X, y = stream.next_sample(max_samples)
    # Trying to predict without fitting
    with pytest.warns(UserWarning):
        learner.predict(X[0])

    learner.partial_fit(X, y)

    expected_description = "if Attribute 11 <= 0.36737233297880056:\n" \
                            "  Leaf = Statistics {0: 450.0000, 1: [-23322.8079, -30257.1616, -18740.9462], " \
                            "2: [22242706.1751, 29895648.2424, 18855571.7943]}\n" \
                            "if Attribute 11 > 0.36737233297880056:\n" \
                            "  Leaf = Statistics {0: 250.0000, 1: [33354.8675, 32390.6094, 22886.4176], " \
                            "2: [15429435.6709, 17908472.4289, 10709746.1079]}\n" \

    assert SequenceMatcher(
        None, expected_description, learner.get_model_description()
    ).ratio() > 0.9


def test_isoup_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    X, y = stream[:, :-2], stream[:, -2:]

    nominal_attr_idx = np.arange(8)
    learner = iSOUPTreeRegressor(
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
    learner = iSOUPTreeRegressor(
        nominal_attributes=nominal_attr_idx,
        leaf_prediction='adaptive'
    )

    learner.partial_fit(X, y)
    learner.predict([new_sample])
