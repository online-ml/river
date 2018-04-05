import numpy as np
from array import array
from skmultiflow.data.generators.random_tree_generator import RandomTreeGenerator
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree


def test_hoeffding_tree():
    stream = RandomTreeGenerator(tree_seed=23, instance_seed=12, n_classes=4, n_nominal_attributes=2,
                                 n_numerical_attributes=5, n_values_per_nominal=5, max_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    nominal_attr_idx = [x for x in range(15, len(stream.get_attributes_header()))]
    learner = HoeffdingTree(nominal_attributes=nominal_attr_idx)

    cnt = 0
    max_samples = 5000
    predictions = array('d')
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_instance()
        # Test every n samples
        if cnt % wait_samples == 0:
            predictions.append(learner.predict(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('d', [0.0, 3.0, 2.0, 1.0, 1.0, 2.0, 0.0, 2.0, 0.0, 3.0,
                                       3.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0,
                                       0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 2.0, 0.0, 0.0,
                                       0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 1.0,
                                       1.0, 3.0, 1.0, 0.0, 3.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    assert np.alltrue(predictions == expected_predictions)

    expected_info = 'HoeffdingTree: max_byte_size: 33554432 - memory_estimate_period: 1000000 - grace_period: 200 ' \
                    '- split_criterion: info_gain - split_confidence: 1e-07 - tie_threshold: 0.05 ' \
                    '- binary_split: False - stop_mem_management: False - remove_poor_atts: False ' \
                    '- no_pre_prune: False - leaf_prediction: nba - nb_threshold: 0 - nominal_attributes: [] - '
    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 1384.0, 1.0: 1720.0, 2.0: 1005.0, 3.0: 891.0}\n'
    expected_model_2 = 'Leaf = Class 1.0 | {1.0: 1720.0, 2.0: 1005.0, 0.0: 1384.0, 3.0: 891.0}\n'
    assert (learner.get_model_description() == expected_model_1) \
           or (learner.get_model_description() == expected_model_2)
