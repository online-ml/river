import math
import sys
import numbers
import copy

from collections import deque

import numpy as np


def get_dimensions(X) -> tuple:
    """Return the dimensions from a numpy.array, numpy.ndarray or list.

    Parameters
    ----------
    X
        numpy.array, numpy.ndarray, list, list of lists.

    Returns
    -------
    A tuple representing the X structure's dimensions.

    """
    r, c = 1, 1
    if isinstance(X, type(np.array([0]))):
        if X.ndim > 1:
            r, c = X.shape
        else:
            r, c = 1, X.size

    elif isinstance(X, type([])):
        if isinstance(X[0], type([])):
            r, c = len(X), len(X[0])
        else:
            c = len(X)

    return r, c


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

    if raise_error and (factor == 0 or math.isnan(factor)):
        raise ValueError(f"Can not normalize, normalization factor is {factor}")

    if not inplace:
        dictionary = copy.deepcopy(dictionary)

    if math.isnan(factor) or factor == 0:
        # Can not normalize, return gracefully
        return dictionary

    for (
        key,
        value,
    ) in dictionary.items():  # loop over the keys, values in the dictionary
        dictionary[key] = value / factor

    return dictionary


def get_max_value_key(dictionary):
    """Get the key of the maximum value in a dictionary.

    Parameters
    ----------
    dictionary
        Dictionary to evaluate.

    Returns
    -------
    int
        Key of the maximum value.

    """
    if dictionary and isinstance(dictionary, dict):
        return max(dictionary, key=dictionary.get)
    else:
        return 0


def calculate_object_size(obj, unit="byte") -> int:
    """Iteratively calculates the `obj` size in bytes.

    Visits all the elements related to obj accounting for their respective
    sizes.

    Parameters
    ----------
    object
        Object to evaluate.
    string
        The unit in which the accounted value is going to be returned.
        Values: 'byte', 'kB', 'MB' (Default: 'byte').

    Returns
    -------
    The size of the object and its related properties and objects, in 'unit'.

    """
    seen = set()
    to_visit = deque()
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
        final_size = byte_size / (2 ** 20)
    else:
        final_size = byte_size

    return final_size


def is_scalar_nan(x) -> bool:
    """Tests if x is NaN

    This function is meant to overcome the issue that np.isnan does not allow
    non-numerical types as input, and that np.nan is not np.float('nan').

    Parameters
    ----------
    x
        any type

    Examples
    --------
    >>> is_scalar_nan(np.nan)
    True
    >>> is_scalar_nan(float("nan"))
    True
    >>> is_scalar_nan(None)
    False
    >>> is_scalar_nan("")
    False
    >>> is_scalar_nan([np.nan])
    False
    """
    # convert from numpy.bool_ to python bool to ensure that testing
    # is_scalar_nan(x) is True does not fail.
    return bool(isinstance(x, numbers.Real) and np.isnan(x))


def add_dict_values(dict_a: dict, dict_b: dict, inplace=False) -> dict:
    """Adds two dictionaries, summing the values of elements with the same key.

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


def add_delay_to_timestamps(timestamps, delay):
    """Add a given delay to a list of timestamps.

    This function iterates over the timestamps, adding a time delay to them.

    Parameters
    ----------
    timestamps
        np.ndarray(dtype=datetime64).
    delay
        np.timedelta64.

    Returns
    -------
    A list of timestamps with a delay added to all timestamp.

    """

    delay_timestamps = []
    for t in timestamps:
        delay_timestamps.append(t + delay)
    return np.array(delay_timestamps, dtype="datetime64")


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
    will be used for rounding; otherwise, decimal places will be added or removed
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
    """
    return round(x, significant_digits - int(math.floor(math.log10(abs(x)))) - 1)
