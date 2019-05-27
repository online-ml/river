import numpy as np
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.lazy import KDTree


def test_kdd_tree_euclidean():
    stream = RandomTreeGenerator(tree_random_state=1, sample_random_state=1)
    stream.prepare_for_use()

    X, _ = stream.next_sample(1000)
    X_test, _ = stream.next_sample(10)

    # Build tree
    kdtree = KDTree(X, metric='euclidean', return_distance=True)

    # Query tree
    dist, idx = kdtree.query(X_test, 4)

    expected_idx = [[855, 466, 348, 996],
                    [829, 654, 92, 333],
                    [227, 364, 183, 325],
                    [439, 482, 817, 501],
                    [886, 173, 279, 470],
                    [98, 30, 34, 580],
                    [959, 773, 374, 819],
                    [819, 685, 59, 992],
                    [624, 665, 209, 239],
                    [524, 807, 506, 191]]
    expected_dist = [[1.6366216258724973, 1.631437068636607, 1.5408182139320563, 1.4836054196064452],
                     [1.7839579422032452, 1.7694587302438618, 1.5339920309706585, 1.5228981881653287],
                     [1.6512443805072872, 1.637456923425164, 1.61736766513639, 1.5776532815820448],
                     [1.5843121606184263, 1.571918014408251, 1.5038147281265382, 0.7058569455034059],
                     [2.052148026638031, 2.0157953468214007, 1.8012794130725434, 1.6572756455115591],
                     [1.5844032729792423, 1.5688736638121885, 1.55893121879858, 1.4609657517960262],
                     [1.6819916227667229, 1.6186557774269037, 1.5815309744477162, 1.5720184136312232],
                     [1.7302164693989817, 1.5964713159009083, 1.4897849225874815, 1.1629448414734906],
                     [1.6511813695220574, 1.6454651930288255, 1.5926685577827064, 1.4973008307362947],
                     [1.5982346741983797, 1.5875900895982191, 1.4702209684850878, 1.4676217546305874]]

    assert np.alltrue(idx == expected_idx)

    assert np.allclose(dist, expected_dist)

    expected_info = 'KDTree(categorical_list=None, leaf_size=40, metric=euclidean, return_distance=True)'
    assert kdtree.get_info() == expected_info

    assert kdtree._estimator_type == 'data_structure'


def test_kdd_tree_mixed():
    stream = RandomTreeGenerator(tree_random_state=1, sample_random_state=1, n_num_features=0)
    stream.prepare_for_use()

    X, _ = stream.next_sample(1000)
    X_test, _ = stream.next_sample(10)

    # Build tree
    cat_features = [i for i in range(25)]
    kdtree = KDTree(X, metric='mixed', return_distance=True, categorical_list=cat_features)

    # Query tree
    dist, idx = kdtree.query(X_test, 4)

    expected_idx = [[123, 234, 707, 654],
                    [688, 429, 216, 627],
                    [463, 970, 566, 399],
                    [18, 895, 640, 996],
                    [396, 612, 897, 232],
                    [328, 54, 138, 569],
                    [253, 501, 82, 273],
                    [38, 146, 752, 923],
                    [946, 808, 271, 363],
                    [951, 111, 708, 5]]
    expected_dist = [[2, 2, 2, 2],
                     [2, 2, 2, 2],
                     [2, 2, 2, 2],
                     [2, 2, 2, 0],
                     [2, 2, 2, 0],
                     [2, 2, 2, 0],
                     [2, 2, 2, 2],
                     [2, 2, 0, 0],
                     [2, 2, 2, 0],
                     [2, 2, 2, 2]]
    assert np.alltrue(idx == expected_idx)

    assert np.allclose(dist, expected_dist)

    expected_info = 'KDTree(categorical_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ' \
                    '11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], ' \
                    'leaf_size=40, metric=mixed, return_distance=True)'
    assert kdtree.get_info() == expected_info

    assert kdtree._estimator_type == 'data_structure'

