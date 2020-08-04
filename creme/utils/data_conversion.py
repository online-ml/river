import numpy as np


def dict2numpy(data):
    """Convert a dictionary containing data to a numpy.array

    Parameters
    ----------
    data : dict
        A dictionary whose keys represent input attributes and the values
        represent their observed contents.

    Returns
    -------
        numpy.array
            An array representation of the values in `data`.

    Examples
    --------
    >>> from creme.utils import dict2numpy
    >>> dict2numpy({'a': 1, 'b': 2, 3: 3})
    array([1, 2, 3])

    Notes
    -----
        There is not restriction to the type of keys in `data`, but values
        must be strictly numeric.
    """
    return np.asarray(list(data.values()))


def numpy2dict(data):
    """Convert a numpy array to a dictionary.

    Parameters
    ----------
    data : numpy.array
        An one-dimensional numpy.array.

    Returns
    -------
        dict
            A dictionary where keys are integers :math: `k \in \{0, 1, ..., |\mathtt{data}| - 1\}`,
            and the values are each one of the :math: `k` entries in `data`.

    Examples
    --------
    >>> import numpy as np
    >>> from creme.utils import numpy2dict
    >>> numpy2dict(np.array([1.0, 2.0, 3.0]))
    {0: 1.0, 1: 2.0, 2: 3.0}
    """
    return {k: v for k, v in enumerate(data)}
