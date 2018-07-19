import numpy as np
from array import array
import os
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HAT


def test_HAT(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=4, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    nominal_attr_idx = [x for x in range(5, stream.n_features)]
    learner = HAT(nominal_attributes=nominal_attr_idx)

    cnt = 0
    max_samples = 5000
    predictions = array('d')
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

    expected_predictions = array('d', [2.0, 1.0, 1.0, 1.0, 0.0, 3.0, 0.0, 1.0, 1.0, 2.0,
                                       0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 1.0,
                                       1.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.0, 3.0, 2.0,
                                       1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0,
                                       0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 2.0])

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree.npy')

    data = np.load(test_file)

    assert np.alltrue(predictions == expected_predictions)
    assert np.allclose(proba_predictions, data)

    expected_info = 'HAT: max_byte_size: 33554432 - memory_estimate_period: 1000000 - grace_period: 200' \
                    ' - split_criterion: info_gain - split_confidence: 1e-07 - tie_threshold: 0.05' \
                    ' - binary_split: False - stop_mem_management: False - remove_poor_atts: False' \
                    ' - no_pre_prune: False - leaf_prediction: nba - nb_threshold: 0' \
                    ' - nominal_attributes: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14] - '

    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 1367.3628584299263, 1.0: 1702.2738590243584,' \
                       ' 2.0: 952.1668539501372, 3.0: 822.1964285955778}\n'
    expected_model_2 = 'Leaf = Class 1.0 | {1.0: 1702.2738590243584, 2.0: 952.1668539501372,' \
                       ' 0.0: 1367.3628584299263, 3.0: 822.1964285955778}\n'
    expected_model_3 = 'Leaf = Class 1.0 | {1.0: 1702.2738590243584, 2.0: 952.16685395013724, ' \
                       '0.0: 1367.3628584299263, 3.0: 822.1964285955778}\n'   # Python 3.6
    expected_model_4 = 'Leaf = Class 1.0 | {0.0: 1367.3628584299263, 1.0: 1702.2738590243584,' \
                       ' 2.0: 952.16685395013724, 3.0: 822.1964285955778}\n'  # Python 3.4

    assert (learner.get_model_description() == expected_model_1) \
           or (learner.get_model_description() == expected_model_2) \
           or (learner.get_model_description() == expected_model_3) \
           or (learner.get_model_description() == expected_model_4)
