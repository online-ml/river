"""
Utilities for streaming data from various sources.
"""
import csv
import datetime as dt
import itertools

from sklearn import utils


__all__ = [
    'iter_csv',
    'iter_numpy',
    'iter_pandas',
    'iter_sklearn_dataset'
]


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
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    feature_names = list(range(len(X[0]))) if feature_names is None else feature_names
    rng = utils.check_random_state(random_state)

    if shuffle:
        order = rng.permutation(len(X))
        X, y = X[order], y if y is None else y[order]

    for x, yi in itertools.zip_longest(X, y if hasattr(y, '__iter__') else []):
        yield {i: xi for i, xi in zip(feature_names, x)}, yi


def iter_sklearn_dataset(load_dataset, **kwargs):
    """Yields rows from one of the datasets provided by scikit-learn.

    Parameters:
        load_dataset (callable): The method used to load the dataset, for example
        `sklearn.datasets.load_boston`.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

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
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    kwargs['feature_names'] = X.columns
    for x, yi in iter_numpy(X, y, **kwargs):
        yield x, yi


def iter_csv(filepath_or_buffer, target_name, types=None, parse_dates=None):
    """Yields rows from a CSV file.

    Parameters:
        filepath_or_buffer: Either a string indicating the location of a CSV file, or a buffer
            object that has a ``read`` method.
        types (dict): The type of each feature.
        parse_dates (dict): A `dict` mapping feature names to a format passed to the
            `datetime.datetime.strptime` method.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    Example:

        ::

            >>> import io
            >>> from creme import stream

            >>> data = io.StringIO('''name,day,viewers
            ... Breaking Bad,2018-03-14,1337
            ... The Sopranos,2018-03-14,42
            ... Breaking Bad,2018-03-15,7331
            ... ''')

            >>> params = dict(
            ...     target_name='viewers',
            ...     types={'viewers': int},
            ...     parse_dates={'day': '%Y-%m-%d'}
            ... )

            >>> for x, y in stream.iter_csv(data, **params):
            ...     print(x, y)
            OrderedDict([('name', 'Breaking Bad'), ('day', datetime.datetime(2018, 3, 14, 0, 0))]) 1337
            OrderedDict([('name', 'The Sopranos'), ('day', datetime.datetime(2018, 3, 14, 0, 0))]) 42
            OrderedDict([('name', 'Breaking Bad'), ('day', datetime.datetime(2018, 3, 15, 0, 0))]) 7331

    """

    file = filepath_or_buffer

    if not hasattr(file, 'read'):
        file = open(file)

    for x in csv.DictReader(file):

        # Cast the values to the given types
        if types is not None:
            for i, t in types.items():
                x[i] = t(x[i])

        # Parse the dates
        if parse_dates is not None:
            for i, fmt in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)

        # Separate the target from the features
        y = x.pop(target_name)

        yield x, y
