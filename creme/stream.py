"""Helper functions for streaming data."""
import csv
import datetime as dt
import functools
import glob
import gzip
import inspect
import io
import itertools
import pickle
import platform
import random
import types
import os
import zipfile

import numpy as np

from . import utils


__all__ = [
    'Cache',
    'iter_array',
    'iter_csv',
    'iter_libsvm',
    'iter_pandas',
    'iter_sklearn_dataset',
    'iter_vaex',
    'shuffle'
]



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


def iter_vaex(X, y=None, features=None, **kwargs):
    """Yields rows from a ``vaex.DataFrame``.

    Parameters:
        X (vaex.DataFrame): A vaex DataFrame housing the training featuers.
        y (string or vaex.Expression): The column or expression containing the target variable.
        features (list of strings or vaex.Expressions): A list of features used for training.
        If None, all columns in ``X`` will be used. Features specifying in ``y`` are ignored.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.
    """

    from vaex.utils import _ensure_strings_from_expressions, _ensure_list

    features = _ensure_strings_from_expressions(features)
    feature_names = features or X.get_column_names()

    if y:
        y = _ensure_strings_from_expressions(y)
        y = _ensure_list(y)
        feature_names = [feat for feat in feature_names if feat not in y]

    multioutput = len(y) > 1

    if multioutput:
        for i in range(len(X)):
            yield (
                {key: X.evaluate(key, i, i + 1)[0] for key in feature_names},
                {key: X.evaluate(key, i, i + 1)[0] for key in y}
            )

    else:

        for i in range(len(X)):
            yield (
                {key: X.evaluate(key, i, i + 1)[0] for key in feature_names},
                X.evaluate(y[0], i, i + 1)[0]
            )


class DictReader(csv.DictReader):
    """Overlay on top of `csv.DictReader` which allows sampling."""

    def __init__(self, fraction, rng, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction = fraction
        self.rng = rng

    def __next__(self):

        if self.line_num == 0:
            self.fieldnames

        row = next(self.reader)

        if self.fraction < 1:
            while self.rng.random() > self.fraction:
                row = next(self.reader)

        return dict(zip(self.fieldnames, row))


def open_filepath(filepath_or_buffer, compression):

    # Determine the compression from the file extension if "infer" has been specified
    if compression == 'infer':
        _, ext = os.path.splitext(filepath_or_buffer)
        compression = {'.gz': 'gzip', '.zip': 'zip'}.get(ext)

    def open_zipfile(path):

        with zipfile.ZipFile(path, 'r') as zf:
            f = zf.open(zf.namelist()[0], 'r')
            f = io.TextIOWrapper(f)
            return f

    # Determine the file opening method from the compression
    open_func = {
        None: open,
        'gzip': functools.partial(gzip.open, mode='rt'),
        'zip': open_zipfile
    }[compression]

    # Open the file using the opening method
    return open_func(filepath_or_buffer)


def iter_csv(filepath_or_buffer, target_name, converters=None, parse_dates=None, drop=None,
             fraction=1., compression='infer', seed=None, field_size_limit=None, **kwargs):
    """Yields rows from a CSV file.

    Parameters:
        filepath_or_buffer: Either a string indicating the location of a CSV file, or a buffer
            object that has a ``read`` method.
        target_name (str): The name of the target.
        converters (dict): A `dict` mapping feature names to callables used to parse their
            associated values.
        parse_dates (dict): A `dict` mapping feature names to a format passed to the
            `datetime.datetime.strptime` method.
        drop (list): Fields to ignore.
        fraction (float): Sampling fraction.
        compression (str): For on-the-fly decompression of on-disk data. If 'infer' and
            ``filepath_or_buffer`` is path-like, then the decompression method is inferred for the
            following extensions: '.gz', '.zip'.
        seed (int): If specified, the sampling will be deterministic.
        field_size_limit (int): If not `None`, this will passed to the `csv.field_size_limit`
            function.

    All other arguments are passed to the underlying `csv.DictReader`.

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
            ...     converters={'viewers': int},
            ...     parse_dates={'day': '%Y-%m-%d'}
            ... )

            >>> for x, y in stream.iter_csv(data, **params):
            ...     print(x, y)
            {'name': 'Breaking Bad', 'day': datetime.datetime(2018, 3, 14, 0, 0)} 1337
            {'name': 'The Sopranos', 'day': datetime.datetime(2018, 3, 14, 0, 0)} 42
            {'name': 'Breaking Bad', 'day': datetime.datetime(2018, 3, 15, 0, 0)} 7331

    """

    # Set the field size limit
    limit = csv.field_size_limit()
    if field_size_limit is not None:
        csv.field_size_limit(field_size_limit)

    # If a file is not opened, then we open it
    if not hasattr(filepath_or_buffer, 'read'):
        filepath_or_buffer = open_filepath(filepath_or_buffer, compression)

    for x in DictReader(
        fraction=fraction,
        rng=random.Random(seed),
        f=filepath_or_buffer,
        **kwargs
    ):

        if drop:
            for i in drop:
                del x[i]

        # Cast the values to the given types
        if converters is not None:
            for i, t in converters.items():
                x[i] = t(x[i])

        # Parse the dates
        if parse_dates is not None:
            for i, fmt in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)

        # Separate the target from the features
        y = x.pop(target_name)

        yield x, y

    # Close the file
    filepath_or_buffer.close()

    # Reset the file size limit to it's original value
    csv.field_size_limit(limit)


def iter_libsvm(filepath_or_buffer, target_type=float, compression='infer'):
    """Iterates over a dataset in LIBSVM format.

    The LIBSVM format is a popular way in the machine learning community to store sparse datasets.
    Only numerical feature values are supported. The feature names will be considered as strings.

    Parameters:
        filepath_or_buffer: Either a string indicating the location of a CSV file, or a buffer
            object that has a ``read`` method.
        target_type (type): Type of the target value.
        compression (str): For on-the-fly decompression of on-disk data. If 'infer' and
            ``filepath_or_buffer`` is path-like, then the decompression method is inferred for the
            following extensions: '.gz', '.zip'.


    Example:

        ::

            >>> import io
            >>> from creme import stream

            >>> data = io.StringIO('''+1 x:-134.26 y:0.2563
            ... 1 x:-12 z:0.3
            ... -1 y:.25
            ... ''')

            >>> for x, y in stream.iter_libsvm(data, target_type=int):
            ...     print(y, x)
            1 {'x': -134.26, 'y': 0.2563}
            1 {'x': -12.0, 'z': 0.3}
            -1 {'y': 0.25}

    References:
        1. `LIBSVM documentation <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`_

    """

    # If a file is not opened, then we open it
    if not hasattr(filepath_or_buffer, 'read'):
        filepath_or_buffer = open_filepath(filepath_or_buffer, compression)

    def split_pair(pair):
        name, value = pair.split(':')
        value = float(value)
        return name, value

    for line in filepath_or_buffer:

        # Remove carriage return and whitespace
        line = line.rstrip()
        # Remove potential end of line comments
        line = line.split('#')[0]

        y, x = line.split(' ', maxsplit=1)
        y = target_type(y)
        x = dict([split_pair(pair) for pair in x.split(' ')])
        yield x, y


def shuffle(stream, buffer_size, seed=None):
    """Shuffles a stream of data.

    This works by maintaining a buffer of elements. The first buffer_size elements are stored in
    memory. Once the buffer is full, a random element inside the buffer is yielded. Everytime an
    element is yielded, the next element in the stream replaces it and the buffer is sampled again.
    Increasing buffer_size will improve the quality of the shuffling.

    If you really want to stream over your dataset in a "good" random order, the best way is to
    split your dataset into smaller datasets and loop over them in a round-robin fashion. You may
    do this by using the ``roundrobin`` recipe from the `itertools` module.

    Example:

        ::

            >>> from creme import stream

            >>> for i in stream.shuffle(range(15), buffer_size=5, seed=42):
            ...     print(i)
            0
            5
            2
            1
            8
            9
            6
            4
            11
            12
            10
            7
            14
            13
            3

    References:
        1. `Visualizing TensorFlow's streaming shufflers <http://www.moderndescartes.com/essays/shuffle_viz/>`_

    """

    rng = random.Random(seed)

    # If stream is not a generator, then we coerce it to one
    if not isinstance(stream, types.GeneratorType):
        stream = iter(stream)

    # Initialize the buffer with the first buffer_size elements of the stream
    buffer = list(itertools.islice(stream, buffer_size))

    # Deplete the stream until it is empty
    for element in stream:

        # Pick a random element from the buffer and yield it
        i = rng.randint(0, len(buffer) - 1)
        yield buffer[i]

        # Replace the yielded element from the buffer with the new element from the stream
        buffer[i] = element

    # Shuffle the remaining buffer elements and yield them one by one
    rng.shuffle(buffer)
    for element in buffer:
        yield element


class Cache:
    """Utility for caching iterables.

    This can be used to save a stream of data to the disk in order to iterate over it faster the
    following time. This can save time depending on the nature of stream. The more processing
    happens in a stream, the more time will be saved. Even in the case where no processing is done
    apart from reading the data, the cache will save some time because it is using the pickle
    binary protocol. It can thus improve the speed in common cases such as reading from a CSV file.

    Parameters:
        directory (str): The path where to store the pickled data streams. If not provided, then
            it will be automatically inferred whenever possible, if not an exception will be
            raised.

    Attributes:
        keys (set): The set of keys that are being cached.

    Example:

        ::

            >>> import time
            >>> from creme import datasets
            >>> from creme import stream

            >>> X_y = datasets.Phishing()
            >>> cache = stream.Cache()

            The cache can be used by wrapping it around an iterable. Because this is the first time
            are iterating over the data, nothing is cached.

            >>> tic = time.time()
            >>> for x, y in cache(X_y, key='phishing'):
            ...     pass
            >>> toc = time.time()
            >>> print(toc - tic)  # doctest: +SKIP
            0.012813

            If we do the same thing again, we can see the loop is now faster.

            >>> tic = time.time()
            >>> for x, y in cache(X_y, key='phishing'):
            ...     pass
            >>> toc = time.time()
            >>> print(toc - tic)  # doctest: +SKIP
            0.001927

            We can see an overview of the cache. The first line indicates the location of the
            cache.

            >>> cache
            /tmp
            phishing - 125.2KiB

            Finally, we can clear the stream from the cache.

            >>> cache.clear('phishing')
            >>> cache
            /tmp

            There is also a ``clear_all`` method to remove all the items in the cache.

            >>> cache.clear_all()

    """

    def __init__(self, directory=None):

        # Guess the directory from the system
        system = platform.system()
        if directory is None:
            directory = {'Linux': '/tmp', 'Darwin': '/tmp'}.get(system)

        if directory is None:
            raise ValueError('There is no default directory defined for {systems} systems, '
                             'please provide one')

        self.directory = directory
        self.keys = set()

        # Check if there is anything already in the cache
        for f in glob.glob(f'{self.directory}/*.creme_cache.pkl'):
            key = os.path.basename(f).split('.')[0]
            self.keys.add(key)

    def _get_path(self, key):
        return os.path.join(self.directory, f'{key}.creme_cache.pkl')

    def __call__(self, stream, key=None):

        # Try to guess a key from the stream object
        if key is None:
            if inspect.isfunction(stream):
                key = stream.__name__

        if key is None:
            raise ValueError('No default key could be guessed for the given stream, '
                             'please provide one')

        path = self._get_path(key)

        if os.path.exists(path):
            yield from self[key]
            return

        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            for el in stream:
                pickler.dump(el)
                yield el
            self.keys.add(key)

    def __getitem__(self, key):
        """Iterates over the stream associated with the given key."""
        with open(self._get_path(key), 'rb') as f:
            unpickler = pickle.Unpickler(f)
            while f.peek(1):
                yield unpickler.load()

    def clear(self, key):
        """Deletes the cached stream associated with the given key."""
        os.remove(self._get_path(key))
        self.keys.remove(key)

    def clear_all(self):
        """Deletes all the cached streams."""
        for key in list(self.keys):
            os.remove(self._get_path(key))
            self.keys.remove(key)

    def __repr__(self):
        return '\n'.join([self.directory] + [
            f'{key} - {utils.pretty.humanize_bytes(os.path.getsize(self._get_path(key)))}'
            for key in self.keys
        ])
