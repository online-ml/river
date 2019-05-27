import os
import numpy as np
from skmultiflow.transform.one_hot_to_categorical import OneHotToCategorical


def test_one_hot_to_categorical(test_path):
    n_categories = 5

    # Load test data generated using:
    # RandomTreeGenerator(tree_random_state=1, sample_random_state=1,
    #                     n_cat_features=n_categories, n_num_features=0)
    test_file = os.path.join(test_path, 'data-one-hot.npz')
    data = np.load(test_file)
    X = data['X']
    y = data['y']

    cat_att_idx = [[i+j for i in range(n_categories)] for j in range(0, n_categories * n_categories, n_categories)]
    transformer = OneHotToCategorical(categorical_list=cat_att_idx)

    X_decoded = transformer.transform(X)

    test_file = os.path.join(test_path, 'data-categorical.npy')
    X_expected = np.load(test_file)
    assert np.alltrue(X_decoded == X_expected)

    X_decoded = transformer.transform(X)
    assert np.alltrue(X_decoded == X_expected)

    expected_info = "OneHotToCategorical(categorical_list=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],\n" \
                    "                                      [10, 11, 12, 13, 14],\n" \
                    "                                      [15, 16, 17, 18, 19],\n" \
                    "                                      [20, 21, 22, 23, 24]])"
    assert transformer.get_info() == expected_info

    assert transformer._estimator_type == 'transform'

    transformer.fit(X=X, y=y)

    transformer.partial_fit_transform(X=X)
