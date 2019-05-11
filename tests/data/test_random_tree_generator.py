import os
import numpy as np
from skmultiflow.data.random_tree_generator import RandomTreeGenerator


def test_random_tree_generator(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ['att_num_0', 'att_num_1', 'att_num_2', 'att_num_3', 'att_num_4',
                       'att_nom_0_val0', 'att_nom_0_val1', 'att_nom_0_val2', 'att_nom_0_val3', 'att_nom_0_val4',
                       'att_nom_1_val0', 'att_nom_1_val1', 'att_nom_1_val2', 'att_nom_1_val3', 'att_nom_1_val4']
    assert stream.feature_names == expected_names

    expected_target_values = [0, 1]
    assert stream.target_values == expected_target_values

    assert stream.target_names == ['class']

    assert stream.n_features == 15

    assert stream.n_cat_features == 2

    assert stream.n_num_features == 5

    assert stream.n_targets == 1

    assert stream.get_data_info() == 'Random Tree Generator - 1 target(s), 2 classes, 15 features'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'random_tree_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    X, y = stream.last_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.n_targets == np.array(y).ndim

    assert stream.n_features == X.shape[1]

    assert 'stream' == stream._estimator_type

    expected_info = "RandomTreeGenerator(fraction_leaves_per_level=0.15, max_tree_depth=6,\n" \
                    "                    min_leaf_depth=3, n_cat_features=2,\n" \
                    "                    n_categories_per_cat_feature=5, n_classes=2,\n" \
                    "                    n_num_features=5, sample_random_state=12,\n" \
                    "                    tree_random_state=23)"
    assert stream.get_info() == expected_info
