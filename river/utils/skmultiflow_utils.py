import copy
import math
import numbers
import sys
import typing
from collections import deque

import numpy as np


def normalize_values_in_dict(dictionary, factor=None, inplace=True, raise_error=False):
    """Normalize the values in a dictionary using the given factor.

    For each element in the dictionary, applies `value/factor`.

    Parameters
    ----------
    dictionary
        Dictionary to normalize.
    factor
        Normalization factor value. If not set, use the sum of values.
    inplace
        If True, perform operation in-place
    raise_error
        In case the normalization factor is either `0` or `None`:</br>
        - `True`: raise an error.
        - `False`: return gracefully (if `inplace=False`, a copy of) `dictionary`.

    Raises
    ------
    ValueError
        In case the normalization factor is either `0` or `None` and `raise_error=True`.

    """
    if factor is None:
        factor = sum(dictionary.values())

    if not inplace:
        dictionary = copy.deepcopy(dictionary)

    if factor == 0 or math.isnan(factor):
        # Can not normalize
        if raise_error:
            raise ValueError(f"Can not normalize, normalization factor is {factor}")
        # return gracefully
        return dictionary

    scale_values_in_dict(dictionary, 1 / factor, inplace=True)

    return dictionary


def scale_values_in_dict(dictionary, multiplier, inplace=True):
    """Scale the values in a dictionary.

    For each element in the dictionary, applies `value * multiplier`.

    Parameters
    ----------
    dictionary
        Dictionary to scale.
    multiplier
        Scaling value.
    inplace
        If True, perform operation in-place

    """

    if not inplace:
        dictionary = copy.deepcopy(dictionary)

    for key, value in dictionary.items():
        dictionary[key] = value * multiplier

    return dictionary


def calculate_object_size(obj: typing.Any, unit: str = "byte") -> int:
    """Iteratively calculates the `obj` size in bytes.

    Visits all the elements related to obj accounting for their respective
    sizes.

    Parameters
    ----------
    obj
        Object to evaluate.
    unit
        The unit in which the accounted value is going to be returned.
        Values: 'byte', 'kB', 'MB' (Default: 'byte').

    Returns
    -------
    The size of the object and its related properties and objects, in 'unit'.

    """
    seen = set()
    to_visit: typing.Deque = deque()
    byte_size = 0

    to_visit.append(obj)

    while True:
        try:
            obj = to_visit.popleft()
        except IndexError:
            break

        # If element was already covered, skip it
        if id(obj) in seen:
            continue

        # Update size accounting
        byte_size += sys.getsizeof(obj)

        # Mark element as seen
        seen.add(id(obj))

        # Add keys and values for size account
        if isinstance(obj, dict):
            for v in obj.values():
                to_visit.append(v)

            for k in obj.keys():
                to_visit.append(k)
        elif hasattr(obj, "__dict__"):
            to_visit.append(obj.__dict__)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
            for i in obj:
                to_visit.append(i)

    if unit == "kB":
        final_size = byte_size / 1024
    elif unit == "MB":
        final_size = byte_size / (2**20)
    else:
        final_size = byte_size

    return int(final_size)


def add_dict_values(dict_a: dict, dict_b: dict, inplace=False) -> dict:
    """Add two dictionaries, summing the values of elements with the same key.

    This function iterates over the keys of dict_b and adds their corresponding
    values to the elements in dict_a. If dict_b has a (key, value) pair that
    does not belong to dict_a, this pair is added to the latter dictionary.

    Parameters
    ----------
    dict_a
        dictionary to update.
    dict_b
        dictionary whose values will be added to `dict_a`.
    inplace
        If `True`, the addition is performed in-place and results are stored in `dict_a`.
        If `False`, `dict_a` is not changed and the results are returned in a new dictionary.

    Returns
    -------
    A dictionary containing the result of the operation. Either a pointer to
    `dict_a` or a new dictionary depending on parameter `inplace`.

    """
    if inplace:
        result = dict_a
    else:
        result = copy.deepcopy(dict_a)

    for k, v in dict_b.items():
        try:
            result[k] += v
        except KeyError:
            result[k] = v
    return result


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Notes
    -----
    Code from sklearn.
    This method is exclusive for cases where np.random is used.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand  # noqa
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f"{seed} cannot be used to seed a numpy.random.RandomState instance")


def round_sig_fig(x, significant_digits=2) -> float:
    """Round considering of significant figures of x, given the select
    `significant_digits` prototype.

    If`significant_digits` match the number of significant figures in `x`, its value
    will be used for rounding; otherwise, decimal places will removed
    accordingly to the significant figures in `x`.

    Parameters
    ----------
    x
        A floating point scalar.
    significant_digits
        The number of intended rounding figures.

    Returns
    -------
        The rounded value of `x`.

    Examples
    --------
    >>> round_sig_fig(1.2345)
    1.2
    >>> round_sig_fig(1.2345, significant_digits=3)
    1.23
    >>> round_sig_fig(0.0)
    0.0
    >>> round_sig_fig(0)
    0
    >>> round_sig_fig(1999, significant_digits=1)
    2000
    >>> round_sig_fig(1999, significant_digits=4)
    1999
    >>> round_sig_fig(0.025, significant_digits=3)
    0.03
    >>> round_sig_fig(0.025, significant_digits=10)
    0.025
    >>> round_sig_fig(0.0250, significant_digits=10)
    0.025
    """
    return round(x, significant_digits - int(math.floor(math.log10(abs(x) + 1))) - 1)
