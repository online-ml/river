import numpy as np
from array import array
import os
from skmultiflow.data import RandomTreeGenerator, SEAGenerator
from skmultiflow.trees import HoeffdingTree


def test_hoeffding_tree_nb(test_path):
    stream = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=4,
        n_cat_features=2, n_num_features=5, n_categories_per_cat_feature=5,
        max_tree_depth=6, min_leaf_depth=3, fraction_leaves_per_level=0.15
    )
    stream.prepare_for_use()

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTree(
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

    expected_info = "HoeffdingTree(binary_split=False, grace_period=200, leaf_prediction='nb',\n" \
                    "              max_byte_size=33554432, memory_estimate_period=1000000,\n" \
                    "              nb_threshold=0, no_preprune=False,\n" \
                    "              nominal_attributes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],\n" \
                    "              remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "              split_criterion='info_gain', stop_mem_management=False,\n" \
                    "              tie_threshold=0.05)"
    assert learner.get_info() == expected_info


def test_hoeffding_tree_nba(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

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

    expected_info = "HoeffdingTree(binary_split=False, grace_period=200, leaf_prediction='nba',\n" \
                    "              max_byte_size=33554432, memory_estimate_period=1000000,\n" \
                    "              nb_threshold=0, no_preprune=False,\n" \
                    "              nominal_attributes=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],\n" \
                    "              remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "              split_criterion='info_gain', stop_mem_management=False,\n" \
                    "              tie_threshold=0.05)"
    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 1423.0, 1.0: 1745.0, 2.0: 978.0, 3.0: 854.0}\n'

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
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) > 0.550 and Att (1) > 0.800 and Att (14) == 0.000 | class: 0\n' + \
        'Att (5) == 1.000 and Att (13) == 0.000 and Att (1) > 0.550 and Att (1) > 0.800 and Att (14) == 1.000 | class: 1\n' + \
        'Att (5) == 1.000 and Att (13) == 1.000 and Att (3) <= 0.730 | class: 1\n' + \
        'Att (5) == 1.000 and Att (13) == 1.000 and Att (3) > 0.730 | class: 0\n'
    assert expected_rules == learner.get_rules_description()


def test_hoeffding_tree_coverage():
    # Cover memory management
    stream = SEAGenerator(random_state=1, noise_percentage=0.05)
    stream.prepare_for_use()
    X, y = stream.next_sample(5000)

    learner = HoeffdingTree(max_byte_size=30, memory_estimate_period=100, grace_period=10, leaf_prediction='mc')

    learner.partial_fit(X, y, classes=stream.target_values)

    learner.reset()

    # Cover nominal attribute observer
    stream = RandomTreeGenerator(tree_random_state=1, sample_random_state=1, n_num_features=0,
                                 n_categories_per_cat_feature=2)
    stream.prepare_for_use()
    X, y = stream.next_sample(1000)
    learner = HoeffdingTree(leaf_prediction='mc', nominal_attributes=[i for i in range(10)])
    learner.partial_fit(X, y, classes=stream.target_values)
