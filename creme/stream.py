"""
Utilities for streaming data from various sources.
"""
from sklearn import utils
import itertools


__all__ = ['iter_numpy', 'iter_pandas', 'iter_sklearn_dataset']


def iter_numpy(X, y=None, feature_names=None, shuffle=False, random_state=None):
    """Yields rows from an array of features and an array of targets.

    Parameters:
        X (array-like of shape (n_samples, n_features))
        y (array-like of shape (n_samples,))
        feature_names (list of length n_features)
        shuffle (bool): Whether to shuffle the inputs or not.
        random_state (int, RandomState instance or None, default=None): If int, ``random_state`` is
            the seed used by the random number generator; if ``RandomState`` instance,
            ``random_state`` is the random number generator; if ``None``, the random number
            generator is the ``RandomState`` instance used by ``np.random``.

    Yields:
        tuple: A pair of (dict, target)

    """
    feature_names = list(range(X.shape[1])) if feature_names is None else feature_names
    rng = utils.check_random_state(random_state)

    if shuffle:
        order = rng.permutation(len(X))
        X, y = X[order], y if y is None else y[order]

    for x, yi in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
        yield {i: xi for i, xi in zip(feature_names, x)}, yi


def iter_sklearn_dataset(load_dataset, **kwargs):
    """Yields rows from one of the datasets provided by scikit-learn.

    Parameters:
        load_dataset (callable): The method used to load the dataset, e.g. ``load_boston``.

    Yields:
        tuple: A pair of (dict, target)

    """
    dataset = load_dataset()
    kwargs['X'] = dataset.data
    kwargs['y'] = dataset.target
    kwargs['feature_names'] = dataset.feature_names

    for x, yi in iter_numpy(**kwargs):
        yield x, yi


def iter_pandas(X, y=None, **kwargs):
    """Yields rows from a ``pandas.DataFrame``.

    Parameters:
        X (pandas.DataFrame)
        y (array-like of shape (n_samples,))

    Yields:
        tuple: A pair of (dict, target)

    """
    kwargs['feature_names'] = X.columns
    for x, yi in iter_numpy(X, y, **kwargs):
        yield x, yi
