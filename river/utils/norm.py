from __future__ import annotations

import copy
import math

__all__ = ["normalize_values_in_dict", "scale_values_in_dict"]


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
