import numpy as np
import math


def get_dimensions(X):
    """ Return the dimensions from a numpy.array, numpy.ndarray or list.

    Parameters
    ----------
    X: numpy.array, numpy.ndarray, list, list of lists.
    
    Returns
    -------
    tuple
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


def normalize_values_in_dict(dictionary, factor=None, inplace=True):
    """ Normalize the values in a dictionary using the given factor.
    For each element in the dictionary, applies ``value/factor``.

    Parameters
    ----------
    dictionary: dict
        Dictionary to normalize.
    factor: float, optional (default=None)
        Normalization factor value. If not set, use the sum of values.
    inplace : bool, default True
        if True, perform operation in-place

    """
    if factor is None:
        factor = sum(dictionary.values())
    if factor == 0:
        raise ValueError('Can not normalize, normalization factor is zero')
    if math.isnan(factor):
        raise ValueError('Can not normalize, normalization factor is NaN')
    if not inplace:
        dictionary = dictionary.copy()
    for key, value in dictionary.items():  # loop over the keys, values in the dictionary
        dictionary[key] = value / factor

    return dictionary


def get_max_value_key(dictionary):
    """ Get the key of the maximum value in a dictionary.

    Parameters
    ----------
    dictionary: dict
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
