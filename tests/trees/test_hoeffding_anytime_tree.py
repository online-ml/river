import numpy as np
from array import array
import os
from skmultiflow.data import RandomTreeGenerator, SEAGenerator
from skmultiflow.trees import HATT


def test_hoeffding_anytime_tree_nb_gini(test_path):
    stream = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=2,
        n_cat_features=2, n_categories_per_cat_feature=4, n_num_features=1,
        max_tree_depth=30, min_leaf_depth=10, fraction_leaves_per_level=0.45
    )
    stream.prepare_for_use()

    learner = HATT(
        nominal_attributes=[i for i in range(1, 9)], leaf_prediction='nb',
        split_criterion='gini'
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

    expected_predictions = array('i', [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
                                       1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0,
                                       1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                       0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)

    expected_info = "HATT(binary_split=False, grace_period=200, leaf_prediction='nb',\n" \
                    "     max_byte_size=33554432, memory_estimate_period=1000000,\n" \
                    "     min_samples_reevaluate=20, nb_threshold=0,\n" \
                    "     nominal_attributes=[1, 2, 3, 4, 5, 6, 7, 8], split_confidence=1e-07,\n" \
                    "     split_criterion='gini', stop_mem_management=False, tie_threshold=0.05)"
    assert learner.get_info() == expected_info


def test_hoeffding_anytime_tree_nba(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_categories_per_cat_feature=4, n_num_features=1, max_tree_depth=30, min_leaf_depth=10,
                                 fraction_leaves_per_level=0.45)
    stream.prepare_for_use()

    learner = HATT(nominal_attributes=[i for i in range(1, 9)])

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

    expected_predictions = array('i',
                                   [1, 0, 1, 0, 1, 1, 1, 0, 0, 0,
                                    0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                                    0, 1, 1, 1, 1, 0, 1, 0, 0, 1,
                                    1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
                                    1, 1, 1, 1, 1, 0, 0, 0, 0])

    assert np.alltrue(predictions == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_anytime_tree.npy')
    expected_proba = np.load(test_file)[:49, :]

    assert np.allclose(proba_predictions, expected_proba)

    expected_info = "HATT(binary_split=False, grace_period=200, leaf_prediction='nba',\n" \
                    "     max_byte_size=33554432, memory_estimate_period=1000000,\n" \
                    "     min_samples_reevaluate=20, nb_threshold=0,\n" \
                    "     nominal_attributes=[1, 2, 3, 4, 5, 6, 7, 8], split_confidence=1e-07,\n" \
                    "     split_criterion='info_gain', stop_mem_management=False,\n" \
                    "     tie_threshold=0.05)"
    assert learner.get_info() == expected_info

    expected_model = 'ifAttribute1=0:ifAttribute3=0:Leaf=Class1|{0:260.0,1:287.0}' \
                     'ifAttribute3=1:Leaf=Class0|{0:163.0,1:117.0}ifAttribute1=1:Leaf=Class0|{0:718.0,1:495.0}'

    assert (learner.get_model_description().replace("\n", " ").replace(" ", "") == expected_model.replace(" ", ""))
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hoeffding_anytime_tree_coverage():
    # Cover memory management
    stream = SEAGenerator(random_state=1, noise_percentage=0.05)
    stream.prepare_for_use()
    X, y = stream.next_sample(5000)

    learner = HATT(max_byte_size=30, memory_estimate_period=100, grace_period=10, leaf_prediction='nba')

    learner.partial_fit(X, y, classes=stream.target_values)

    learner.reset()

    # Cover nominal attribute observer
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_categories_per_cat_feature=4, n_num_features=1, max_tree_depth=30, min_leaf_depth=10,
                                 fraction_leaves_per_level=0.45)
    stream.prepare_for_use()
    X, y = stream.next_sample(5000)
    learner = HATT(leaf_prediction='nba', nominal_attributes=[i for i in range(1, 9)])
    learner.partial_fit(X, y, classes=stream.target_values)
