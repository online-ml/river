import numpy as np
from array import array
import os
from skmultiflow.data import RandomTreeGenerator, SEAGenerator
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
from skmultiflow.utils import calculate_object_size


def test_extremely_fast_decision_tree_nb_gini(test_path):
    stream = RandomTreeGenerator(
        tree_random_state=23, sample_random_state=12, n_classes=2,
        n_cat_features=2, n_categories_per_cat_feature=4, n_num_features=1,
        max_tree_depth=30, min_leaf_depth=10, fraction_leaves_per_level=0.45
    )

    learner = ExtremelyFastDecisionTreeClassifier(
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

    expected_info = "ExtremelyFastDecisionTreeClassifier(binary_split=False, grace_period=200, " \
                    "leaf_prediction='nb', max_byte_size=33554432, memory_estimate_period=1000000, " \
                    "min_samples_reevaluate=20, nb_threshold=0, nominal_attributes=[1, 2, 3, 4, 5, 6, 7, 8], " \
                    "split_confidence=1e-07, split_criterion='gini', stop_mem_management=False, tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info


def test_extremely_fast_decision_tree_nba(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_categories_per_cat_feature=4, n_num_features=1, max_tree_depth=30, min_leaf_depth=10,
                                 fraction_leaves_per_level=0.45)

    learner = ExtremelyFastDecisionTreeClassifier(nominal_attributes=[i for i in range(1, 9)])

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

    expected_info = "ExtremelyFastDecisionTreeClassifier(binary_split=False, grace_period=200, " \
                    "leaf_prediction='nba', max_byte_size=33554432, memory_estimate_period=1000000, " \
                    "min_samples_reevaluate=20, nb_threshold=0, nominal_attributes=[1, 2, 3, 4, 5, 6, 7, 8], " \
                    "split_confidence=1e-07, split_criterion='info_gain', stop_mem_management=False, " \
                    "tie_threshold=0.05)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    expected_model = 'ifAttribute1=0.0:ifAttribute3=0.0:Leaf=Class1|{0:260.0,1:287.0}' \
                     'ifAttribute3=1.0:Leaf=Class0|{0:163.0,1:117.0}ifAttribute1=1.0:Leaf=Class0|{0:718.0,1:495.0}'

    assert (learner.get_model_description().replace("\n", " ").replace(" ", "") == expected_model.replace(" ", ""))
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_extremely_fast_decision_tree_coverage():
    # Cover memory management
    max_size_kb = 20
    stream = SEAGenerator(random_state=1, noise_percentage=0.05)
    X, y = stream.next_sample(5000)

    # Unconstrained model has over 50 kB
    learner = ExtremelyFastDecisionTreeClassifier(
        leaf_prediction='mc', memory_estimate_period=200, max_byte_size=max_size_kb*2**10,
        min_samples_reevaluate=2500
    )

    learner.partial_fit(X, y, classes=stream.target_values)
    assert calculate_object_size(learner, 'kB') <= max_size_kb

    learner.reset()

    # Cover nominal attribute observer
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_categories_per_cat_feature=4, n_num_features=1, max_tree_depth=30, min_leaf_depth=10,
                                 fraction_leaves_per_level=0.45)
    X, y = stream.next_sample(5000)
    learner = ExtremelyFastDecisionTreeClassifier(leaf_prediction='nba', nominal_attributes=[i for i in range(1, 9)])
    learner.partial_fit(X, y, classes=stream.target_values)
