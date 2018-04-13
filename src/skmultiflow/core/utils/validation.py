"""Utilities for input validation."""

import numbers
import numpy as np


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
    Code from sklearn

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState instance'.format(seed))


def check_weights(weight, expand_length=1):
    """Check if weights are valid

    Parameters
    ----------
    weight : int, float, list, np.ndarray
        If weight is a number, returns it inside an np.ndarray
        If weight is a list or np.ndarray, returns it
        Otherwise raise ValueError.
    expand_length : int, optional (default=1)
        If the value passed is larger than 1 and weight is a single value, then the weight is replicated n times inside
        an np.array. If weight is not a single value, raises an error

    """
    if isinstance(weight, (int, float, np.integer, np.float)):
        if expand_length >= 1:
            return np.array([weight] * expand_length, dtype=np.float)
    elif isinstance(weight, list):
        if all(isinstance(x, (int, float, np.integer, np.float)) for x in weight):
            if expand_length == 1:
                return weight
    if isinstance(weight, np.ndarray):
        if weight.size > 1 and all(isinstance(x, (int, float, np.integer, np.float)) for x in weight):
            if expand_length == 1:
                return weight
        elif weight.size == 1 and isinstance(weight[0], (int, float, np.integer, np.float)):
            if expand_length == 1:
                return weight
            elif expand_length > 1:
                return np.array([weight] * expand_length, dtype=np.float)
    raise ValueError('Invalid weight(s): {}'.format(weight))
