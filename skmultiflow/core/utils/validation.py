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


def check_weights(weight):
    """Check if weights are valid
    Parameters
    ----------
    weight : int, float, list, np.ndarray
        If weight is a number (int, float), returns it inside a np.array
        If weight is a list of numbers, returns it
        Otherwise raise ValueError.
    """
    if isinstance(weight, (list, np.ndarray)):
        if all(isinstance(x, (int, float)) for x in weight):
            return weight
    elif isinstance(weight, (int, float, np.integer, np.float)):
        return np.array([weight], dtype=np.float)
    else:
        raise ValueError('Invalid weight(s): {}'.format(weight))
