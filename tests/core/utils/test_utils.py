import numpy as np
from copy import copy
from skmultiflow.core.utils.utils import get_dimensions
from skmultiflow.core.utils.utils import get_max_value_key
from skmultiflow.core.utils.utils import normalize_values_in_dict


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
    for k in range(10):
        a_dictionary[k] = k*10

    reference = copy(a_dictionary)
    sum_of_values = sum(a_dictionary.values())

    normalize_values_in_dict(a_dictionary)
    for k, v in a_dictionary.items():
        assert a_dictionary[k] == reference[k] / sum_of_values

    normalize_values_in_dict(a_dictionary, factor=1/sum_of_values)
    for k, v in a_dictionary.items():
        assert a_dictionary[k] == reference[k]
