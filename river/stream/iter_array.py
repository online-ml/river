import itertools
import random
import typing

import numpy as np

from river import base


def iter_array(
    X: np.ndarray,
    y: np.ndarray = None,
    feature_names: typing.List[base.typing.FeatureName] = None,
    target_names: typing.List[base.typing.FeatureName] = None,
    shuffle: bool = False,
    seed: int = None,
) -> base.typing.Stream:
    """Iterates over the rows from an array of features and an array of targets.

    This method is intended to work with `numpy` arrays, but should also work with Python lists.

    Parameters
    ----------
    X
        A 2D array of features.
    y
        An optional array of targets.
    feature_names
        An optional list of feature names. The features will be labeled with integers if no names
        are provided.
    target_names
        An optional list of output names. The outputs will be labeled with integers if no names are
        provided. Only applies if there are multiple outputs, i.e. if `y` is a 2D array.
    shuffle
        Indicates whether or not to shuffle the input arrays before iterating over them.
    seed
        Random seed used for shuffling the data.

    Examples
    --------

    >>> from river import stream
    >>> import numpy as np

    >>> X = np.array([[1, 2, 3], [11, 12, 13]])
    >>> Y = np.array([True, False])

    >>> dataset = stream.iter_array(
    ...     X, Y,
    ...     feature_names=['x1', 'x2', 'x3']
    ... )
    >>> for x, y in dataset:
    ...     print(x, y)
    {'x1': 1, 'x2': 2, 'x3': 3} True
    {'x1': 11, 'x2': 12, 'x3': 13} False

    """
    feature_names = list(range(len(X[0]))) if feature_names is None else feature_names

    multioutput = y is not None and not np.isscalar(y[0])
    if multioutput and target_names is None:
        target_names = list(range(len(y[0])))

    # Shuffle the data
    rng = random.Random(seed)
    if shuffle:
        order = rng.sample(range(len(X)), k=len(X))
        X = X[order]
        y = y if y is None else y[order]

    if multioutput:

        for xi, yi in itertools.zip_longest(X, y if hasattr(y, "__iter__") else []):
            yield dict(zip(feature_names, xi)), dict(zip(target_names, yi))

    else:

        for xi, yi in itertools.zip_longest(X, y if hasattr(y, "__iter__") else []):
            yield dict(zip(feature_names, xi)), yi
