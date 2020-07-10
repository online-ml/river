import numpy as np
from skmultiflow.data import make_logical


def test_make_logical():
    X, y = make_logical(n_tiles=2, random_state=1)

    expected_X = np.array([[1., 1.],
                           [1., 0.],
                           [0., 1.],
                           [1., 0.],
                           [0., 0.],
                           [0., 0.],
                           [1., 1.],
                           [0., 1.]], dtype=int)
    assert np.alltrue(X == expected_X)

    expected_y = np.array([[1., 0., 1.],
                           [1., 1., 0.],
                           [1., 1., 0.],
                           [1., 1., 0.],
                           [0., 0., 0.],
                           [0., 0., 0.],
                           [1., 0., 1.],
                           [1., 1., 0.]], dtype=int)
    assert np.alltrue(y == expected_y)
