import sys
from copy import copy

import numpy as np

from skmultiflow.utils.utils import get_dimensions
from skmultiflow.utils.utils import get_max_value_key
from skmultiflow.utils.utils import normalize_values_in_dict
from skmultiflow.utils.utils import calculate_object_size


def test_get_dimensions():
    rows_expected = 5
    cols_expected = 5

    a_list = [None] * cols_expected
    rows, cols = get_dimensions(a_list)
    assert rows == 1
    assert cols == cols_expected

    a_list_of_lists = [a_list] * rows_expected
    rows, cols = get_dimensions(a_list_of_lists)
    assert rows == rows_expected
    assert cols == cols_expected

    a_ndarray = np.ndarray(cols_expected)
    rows, cols = get_dimensions(a_ndarray)
    assert rows == 1
    assert cols == cols_expected

    a_ndarray = np.ndarray((rows_expected, cols_expected))
    rows, cols = get_dimensions(a_ndarray)
    assert rows == rows_expected
    assert cols == cols_expected


def test_get_max_value_key():
    a_dictionary = {1: 10, 2: -10, 3: 1000, 4: 100, 5: 1}

    key_max = get_max_value_key(a_dictionary)

    assert key_max == 3


def test_normalize_values_in_dict():
    a_dictionary = {}
    for k in range(1, 11):
        a_dictionary[k] = k*10

    reference = copy(a_dictionary)
    sum_of_values = sum(a_dictionary.values())

    normalize_values_in_dict(a_dictionary)
    for k, v in a_dictionary.items():
        assert a_dictionary[k] == reference[k] / sum_of_values

    normalize_values_in_dict(a_dictionary, factor=1/sum_of_values)
    for k, v in a_dictionary.items():
        assert np.isclose(a_dictionary[k], reference[k])

    b_dictionary = normalize_values_in_dict(a_dictionary, factor=1 / sum_of_values, inplace=False)
    for k, v in a_dictionary.items():
        assert a_dictionary[k] != b_dictionary[k]
    assert id(a_dictionary) != id(b_dictionary)


def test_calculate_object_size():
    elems = []
    array_length = 10

    for i in range(100):
        elems.append(np.ones((array_length), np.int8))
        elems.append('testing_string')

    if sys.platform == 'linux':
        # assert sizes based on a linux system
        assert calculate_object_size(elems, 'byte') == 37335
        assert calculate_object_size(elems, 'kB') == 36.4599609375
        assert calculate_object_size(elems, 'MB') == 0.035605430603027344
    else:
        # run for coverage
        calculate_object_size(elems, 'byte')
        calculate_object_size(elems, 'kB')
        calculate_object_size(elems, 'MB')
