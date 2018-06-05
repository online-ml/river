import numpy as np
from array import array
import os
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


def test_hoeffding_tree(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    cnt = 0
    max_samples = 5000
    predictions = array('d')
    proba_predictions = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if cnt % wait_samples == 0:
            predictions.append(learner.predict(X)[0])
            proba_predictions.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('d', [0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 3.0, 0.0, 1.0, 1.0,
                                       2.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0,
                                       1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0,
                                       2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                                       2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 2.0])

    test_file = os.path.join(test_path, 'test_hoeffding_tree.npz')

    data = np.load(test_file)
    expected_proba_predictions_0 = data["a"]
    expected_proba_predictions_1 = data["b"]

    assert np.alltrue(predictions == expected_predictions)

    assert np.alltrue(proba_predictions == expected_proba_predictions_0) or \
           np.alltrue(proba_predictions == expected_proba_predictions_1)
    assert np.alltrue(predictions == expected_predictions)

    expected_info = 'HoeffdingTree: max_byte_size: 33554432 - memory_estimate_period: 1000000 - grace_period: 200 ' \
                    '- split_criterion: info_gain - split_confidence: 1e-07 - tie_threshold: 0.05 ' \
                    '- binary_split: False - stop_mem_management: False - remove_poor_atts: False ' \
                    '- no_pre_prune: False - leaf_prediction: nba - nb_threshold: 0 - nominal_attributes: [5, 6, 7,' \
                    ' 8, 9, 10, 11, 12, 13, 14] - '
    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 1423.0, 1.0: 1745.0, 2.0: 978.0, 3.0: 854.0}\n'
    expected_model_2 = 'Leaf = Class 1.0 | {1.0: 1745.0, 2.0: 978.0, 0.0: 1423.0, 3.0: 854.0}\n'
    assert (learner.get_model_description() == expected_model_1) \
           or (learner.get_model_description() == expected_model_2)
