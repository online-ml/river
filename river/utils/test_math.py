from __future__ import annotations

import numpy as np

from river import utils


def test_dotvecmat_zero_vector_times_matrix_of_ones():
    A_numpy = np.array([[1, 1], [1, 1]])
    A_river = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    x_numpy = np.array([0, 0])
    x_river = {0: 0, 1: 0}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_vector_of_ones_times_zero_matrix():
    A_numpy = np.array([[0, 0], [0, 0]])
    A_river = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    x_numpy = np.array([1, 1])
    x_river = {0: 1, 1: 1}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_filter_first_matrix_row_with_vector():
    A_numpy = np.array([[1, 1], [1, 1]])
    A_river = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    x_numpy = np.array([1, 0])
    x_river = {0: 1, 1: 0}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_filter_second_matrix_row_with_vector():
    A_numpy = np.array([[1, 1], [1, 1]])
    A_river = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    x_numpy = np.array([0, 1])
    x_river = {0: 0, 1: 1}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_vector_times_identity_matrix():
    A_numpy = np.array([[1, 0], [0, 1]])
    A_river = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    x_numpy = np.array([1, 1])
    x_river = {0: 1, 1: 1}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_vector_times_anti_diagonal_identity_matrix():
    A_numpy = np.array([[0, 1], [1, 0]])
    A_river = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    x_numpy = np.array([1, 1])
    x_river = {0: 1, 1: 1}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]


def test_dotvecmat_three_dimensional_vector_times_non_quadratic_matrix():
    A_numpy = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    A_river = {
        (0, 0): 1,
        (0, 1): 2,
        (0, 2): 3,
        (0, 3): 4,
        (0, 5): 5,
        (1, 0): 6,
        (1, 1): 7,
        (1, 2): 8,
        (1, 3): 9,
        (1, 5): 10,
        (2, 0): 11,
        (2, 1): 12,
        (2, 2): 13,
        (2, 3): 14,
        (2, 5): 15,
    }
    x_numpy = np.array([1, 2, 3])
    x_river = {0: 1, 1: 2, 2: 3}

    numpy_dotvecmat = x_numpy.dot(A_numpy)
    river_dotvecmat = utils.math.dotvecmat(x_river, A_river)

    assert numpy_dotvecmat[0] == river_dotvecmat[0]
    assert numpy_dotvecmat[1] == river_dotvecmat[1]
