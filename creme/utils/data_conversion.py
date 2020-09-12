import numpy as np


def dict2numpy(data) -> np.ndarray:
    """Convert a dictionary containing data to a numpy array.

    There is not restriction to the type of keys in `data`, but values must be strictly numeric.

    Parameters
    ----------
    data
        A dictionary whose keys represent input attributes and the values
        represent their observed contents.

    Returns
    -------
    An array representation of the values in `data`.

    Examples
    --------
    >>> from creme.utils import dict2numpy
    >>> dict2numpy({'a': 1, 'b': 2, 3: 3})
    array([1, 2, 3])

    """
    return np.asarray(list(data.values()))


def numpy2dict(data: np.ndarray) -> dict:
    """Convert a numpy array to a dictionary.

    Parameters
    ----------
    data
        An one-dimensional numpy.array.

    Returns
    -------
    A dictionary where keys are integers $k \in \{0, 1, ..., |\text{data}| - 1\}$,
    and the values are each one of the $k$ entries in `data`.

    Examples
    --------
    >>> import numpy as np
    >>> from creme.utils import numpy2dict
    >>> numpy2dict(np.array([1.0, 2.0, 3.0]))
    {0: 1.0, 1: 2.0, 2: 3.0}

    """
    return {k: v for k, v in enumerate(data)}
