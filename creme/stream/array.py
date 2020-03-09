import itertools
import random

import numpy as np


def iter_array(X, y=None, feature_names=None, target_names=None, shuffle=False, seed=None):
    """Yields rows from an array of features and an array of targets.

    This method is compatible with ``numpy`` arrays as well as Python lists.

    Parameters:
        X (array-like of shape (n_samples, n_features))
        y (array-like of shape (n_samples,))
        feature_names (list of length n_features)
        target_names (list of length n_outputs)
        shuffle (bool): Whether to shuffle the inputs or not.
        seed (int): Random seed used for shuffling the data.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    feature_names = list(range(len(X[0]))) if feature_names is None else feature_names

    multioutput = y is not None and not np.isscalar(y[0])
    if multioutput and target_names is None:
        target_names = list(range(len(y[0])))

    # Shuffle the data
    rng = random.Random(seed)
    if shuffle:
        order = rng.sample(range(len(X)), k=len(X))
        X, y = X[order], y if y is None else y[order]

    if multioutput:

        for x, yi in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
            yield dict(zip(feature_names, x)), dict(zip(target_names, yi))

    else:

        for x, yi in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
            yield dict(zip(feature_names, x)), yi



def iter_pandas(X, y=None, **kwargs):
    """Yields rows from a ``pandas.DataFrame``.

    Parameters:
        X (pandas.DataFrame)
        y (array-like of shape (n_samples,))

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    import pandas as pd

    kwargs['feature_names'] = X.columns
    if isinstance(y, pd.DataFrame):
        kwargs['target_names'] = y.columns

    for x, yi in iter_array(X.to_numpy(), y, **kwargs):
        yield x, yi


def iter_sklearn_dataset(dataset, **kwargs):
    """Yields rows from one of the datasets provided by scikit-learn.

    Parameters:
        dataset (sklearn.utils.Bunch): A scikit-learn dataset.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    kwargs['X'] = dataset.data
    kwargs['y'] = dataset.target
    try:
        kwargs['feature_names'] = dataset.feature_names
    except AttributeError:
        pass

    for x, yi in iter_array(**kwargs):
        yield x, yi
