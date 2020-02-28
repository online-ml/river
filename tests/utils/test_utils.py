import sys
from copy import copy

import numpy as np

from skmultiflow.utils.utils import get_dimensions
from skmultiflow.utils.utils import get_max_value_key
from skmultiflow.utils.utils import normalize_values_in_dict
from skmultiflow.utils.utils import calculate_object_size
from skmultiflow.utils.utils import add_dict_values


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

    if sys.platform == 'linux' and sys.version_info[:2] >= (3, 6):
        # object sizes vary across architectures and OSs
        # following are "expected" sizes for Python 3.6+ on linux systems
        expected_size_in_bytes_1 = 37335
        expected_size_in_bytes_2 = 37343
        expected_size_in_bytes_3 = 37327
        assert np.isclose(calculate_object_size(elems, 'byte'), expected_size_in_bytes_1) or \
               np.isclose(calculate_object_size(elems, 'byte'), expected_size_in_bytes_2) or \
               np.isclose(calculate_object_size(elems, 'byte'), expected_size_in_bytes_3)
    else:
        # only run for coverage
        calculate_object_size(elems, 'byte')

    # Run for coverage the 'kB' and 'MB' variants.
    # No asert is needed since they are based on the 'byte' size.
    calculate_object_size(elems, 'kB')
    calculate_object_size(elems, 'MB')


def test_add_dict_values():
    a = {0: 1, 2: 1}
    b = {1: 1, 2: 1, 3: 0}

    c = add_dict_values(a, b, inplace=False)

    expected = {0: 1, 1: 1, 2: 2, 3: 0}

    for k in c:
        assert c[k] == expected[k]

    add_dict_values(a, b, inplace=True)

    for k in a:
        assert a[k] == expected[k]
