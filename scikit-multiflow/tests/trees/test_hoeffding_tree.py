import numpy as np
from array import array
import os
from skmultiflow.data import RandomTreeGenerator, SEAGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import calculate_object_size


def test_hoeffding_tree_nb(test_path):
    stream = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=4,
        n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
        max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15
    )

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTreeClassifier(
        nominal_attributes=nominal_attr_idx, leaf_prediction='nb'
    )

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1
    expected_predictions = array('i', [0, 1, 3, 0, 0, 3, 0, 1, 1, 2, 0, 2, 1,
                                       1, 2, 1, 3, 0, 1, 1, 1, 1, 0, 3, 1, 2,
                                       1, 1, 3, 2, 1, 2, 2, 2, 1, 1, 1, 0, 1,
                                       2, 0, 2, 0, 0, 0, 0, 1, 3, 2])

    assert np.alltrue(predictions == expected_predictions)

    expected_info = "HoeffdingTreeClassifier(binary_split=False, grace_period=200, leaf_prediction='nb', " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_hoeffding_tree_nba(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    cnt = 0
    max_samples = 5000
    predictions = array('i')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [0, 1, 3, 0, 0, 3, 0, 1, 1, 2,
                                       0, 2, 1, 1, 2, 1, 3, 0, 1, 1,
                                       1, 1, 0, 3, 1, 2, 1, 1, 3, 2,
                                       1, 2, 2, 2, 1, 1, 1, 0, 1, 2,
                                       0, 2, 0, 0, 0, 0, 1, 3, 2])

    test_file = os.path.join(test_path, 'test_hoeffding_tree.npy')

    data = np.load(test_file)

    assert np.alltrue(predictions == expected_predictions)
    assert np.allclose(proba_predictions, data)

    expected_info = "HoeffdingTreeClassifier(binary_split=False, grace_period=200, leaf_prediction='nba', " \
                    "max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0, no_preprune=False, " \
                    "nominal_attributes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], remove_poor_atts=False, " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    expected_model_1 = 'Leaf = Class 1 | {0: 1423.0, 1: 1745.0, 2: 978.0, 3: 854.0}\n'

    assert (learner.get_model_description() == expected_model_1)
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    X, y = stream.next_sample(20000)
    learner.split_criterion = 'hellinger'
    learner.partial_fit(X, y)

    expected_rules = 'Att (5) == 0.000 and Att (12) == 0.000 | class: 1\n' + \
        'Att (5) == 0.000 and Att (12) == 1.000 | class: 1\n' + \
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) <= 0.550 and Att (3) <= 0.730 | class: 0\n' +\
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) <= 0.550 and Att (3) > 0.730 | class: 2\n' + \
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) > 0.550 and Att (1) <= 0.800 | class: 0\n' + \
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) > 0.550 and Att (1) > 0.800 and Att (14) == 0.000' \
        ' | class: 0\n' + \
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) > 0.550 and Att (1) > 0.800 and Att (14) == 1.000' \
        ' | class: 1\n' + \
        'Att (5) == 1.000 and Att (13) == 1.000 and Att (3) <= 0.730 | class: 1\n' + \
        'Att (5) == 1.000 and Att (13) == 1.000 and Att (3) > 0.730 | class: 0\n'
    assert expected_rules == learner.get_rules_description()


def test_hoeffding_tree_coverage():
    # Cover memory management
    max_samples = 5000
    max_size_kb = 50
    stream = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=10,
        n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
        max_tree_depth=15, min_leaf_depth=3, fraction_leaves_per_level=0.15
    )

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    # Unconstrained model has over 72 kB
    learner = HoeffdingTreeClassifier(
        nominal_attributes=nominal_attr_idx, leaf_prediction='mc', memory_estimate_period=100,
        max_byte_size=max_size_kb*2**10
    )

    X, y = stream.next_sample(max_samples)
    learner.partial_fit(X, y)

    assert calculate_object_size(learner, 'kB') <= max_size_kb

    learner.reset()


def test_hoeffding_tree_model_information():
    stream = SEAGenerator(random_state=1, noise_percentage=0.05)
    X, y = stream.next_sample(5000)

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    learner.partial_fit(X, y, classes=stream.target_values)

    expected_info = {
        'Tree size (nodes)': 5,
        'Tree size (leaves)': 3,
        'Active learning nodes': 3,
        'Tree depth': 2,
        'Active leaf byte size estimate': 0.0,
        'Inactive leaf byte size estimate': 0.0,
        'Byte size estimate overhead': 1.0
    }

    observed_info = learner.model_measurements
    for k in expected_info:
        assert k in observed_info
        assert expected_info[k] == observed_info[k]

    expected_description = "if Attribute 0 <= 4.549969620513424:\n" \
                            "  if Attribute 1 <= 5.440182925299016:\n" \
                            "    Leaf = Class 0 | {0: 345.54817975126275, 1: 44.43855503614928}\n" \
                            "  if Attribute 1 > 5.440182925299016:\n" \
                            "    Leaf = Class 1 | {0: 54.451820248737235, 1: 268.5614449638507}\n" \
                            "if Attribute 0 > 4.549969620513424:\n" \
                            "  Leaf = Class 1 | {0: 390.5845685762964, 1: 2372.3747376855454}\n" \

    assert expected_description == learner.get_model_description()


def test_hoeffding_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last two columns (regression targets)
    stream = stream[:, :-2]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = np.arange(7).tolist()
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    learner.partial_fit(X, y, classes=np.unique(y))

    expected_description = "if Attribute 0 = -15.0:\n" \
                           "  Leaf = Class 2 | {2: 350.0}\n" \
                           "if Attribute 0 = 0.0:\n" \
                           "  Leaf = Class 0 | {0: 420.0, 1: 252.0}\n" \
                           "if Attribute 0 = 1.0:\n" \
                           "  Leaf = Class 1 | {0: 312.0, 1: 332.0}\n" \
                           "if Attribute 0 = 2.0:\n" \
                           "  Leaf = Class 1 | {0: 236.0, 1: 383.0}\n" \
                           "if Attribute 0 = 3.0:\n" \
                           "  Leaf = Class 1 | {0: 168.0, 1: 459.0}\n" \
                           "if Attribute 0 = -30.0:\n" \
                           "  Leaf = Class 3 | {3: 46.0, 4: 42.0}\n"

    assert learner.get_model_description() == expected_description
