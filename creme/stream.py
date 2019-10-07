"""Streaming data utilities."""
import csv
import datetime as dt
import functools
import gzip
import itertools
import random
import types
import os

import numpy as np
from sklearn import utils

__all__ = [
    'iter_csv',
    'iter_array',
    'iter_pandas',
    'iter_sklearn_dataset',
    'iter_vaex',
    'simulate_qa',
    'shuffle'
]


def iter_array(X, y=None, feature_names=None, target_names=None, shuffle=False, random_state=None):
    """Yields rows from an array of features and an array of targets.

    This method is compatible with ``numpy`` arrays as well as Python lists.

    Parameters:
        X (array-like of shape (n_samples, n_features))
        y (array-like of shape (n_samples,))
        feature_names (list of length n_features)
        target_names (list of length n_outputs)
        shuffle (bool): Whether to shuffle the inputs or not.
        random_state (int, np.random.RandomState instance or None, default=None): If int,
            ``random_state`` is the seed used by the random number generator; if ``RandomState``
            instance, ``random_state`` is the random number generator; if ``None``, the random
            number generator is the ``RandomState`` instance used by ``np.random``.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    """
    feature_names = list(range(len(X[0]))) if feature_names is None else feature_names

    multioutput = y is not None and not np.isscalar(y[0])
    if multioutput and target_names is None:
        target_names = list(range(len(y[0])))

    # Shuffle the data
    rng = utils.check_random_state(random_state)
    if shuffle:
        order = rng.permutation(len(X))
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

        self.line_num = self.reader.line_num

        while row == []:
            row = next(self.reader)
        d = dict(zip(self.fieldnames, row))
        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            d[self.restkey] = row[lf:]
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d


def iter_csv(filepath_or_buffer, target_name, names=None, converters=None, parse_dates=None, fraction=1.,
             compression='infer', seed=None):
    """Yields rows from a CSV file.

    Parameters:
        filepath_or_buffer: Either a string indicating the location of a CSV file, or a buffer
            object that has a ``read`` method.
        target_name (str): The name of the target.
        names (list of str): A list of names to associate with each element in a row. If ``None``,
            then the first row will be assumed to contain the names.
        converters (dict): A `dict` mapping feature names to callables used to parse their
            associated values.
        parse_dates (dict): A `dict` mapping feature names to a format passed to the
            `datetime.datetime.strptime` method.
        fraction (float): Sampling fraction.
        compression (str): For on-the-fly decompression of on-disk data. If 'infer' and
            ``filepath_or_buffer`` is path-like, then the decompression method is inferred for the
            following extensions: '.gz'.
        seed (int): If specified, the sampling will be deterministic.

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

    # If a file is not opened, then we open it
    if not hasattr(filepath_or_buffer, 'read'):

        # Determine the compression from the file extension if "infer" has been specified
        if compression == 'infer':
            _, ext = os.path.splitext(filepath_or_buffer)
            compression = {
                '.csv': 'csv',
                '.gz': 'gzip'
            }[ext]

        # Determine the file opening method from the compression
        open_func = {
            'csv': open,
            'gzip': functools.partial(gzip.open, mode='rt')
        }[compression]

        # Open the file using the opening method
        filepath_or_buffer = open_func(filepath_or_buffer)

    for x in DictReader(
        fraction=fraction,
        rng=random.Random(seed),
        f=filepath_or_buffer,
        fieldnames=names
    ):

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


def simulate_qa(X_y, on, lag):
    """Simulate a real case learning scenario using a temporal attribute and a lag.

    Each observation will first be shown without revealing the ``y`` value. Once the observation is
    old enough it will be shown once again, this time with the ``y`` value being revealed. Each
    observation will thus be shown twice, once without the target equal to ``None`` and once with
    the actual value. The duration between "questions" and "answers" depends on the ``lag``
    parameter. For this to work, the data is assumed to be sorted with respect to the temporal
    attribute.

    Parameters:
        X_y (generator): A stream of (``x``, ``y``) pairs.
        on (str): The attribute used for measuring time.
        lag (datetime.timedelta or int or float): Amount to wait before revealing the target
            associated with each observation. This value is expected to be able to sum with the
            ``on`` attribute.

    Yields:
        (``q_a``, ``x``, ``y``) pairs where ``q_a`` is a `bool` that indicates if the current
            iteration is question or not.

    Example:

        >>> from creme import stream

        >>> X_y = [({'moment': i, 'x': i}, bool(i % 2)) for i in range(5)]

        >>> for is_question, x, y in simulate_qa(X_y, on='moment', lag=3):
        ...     print('Q' if is_question else 'A', x, y)
        Q {'moment': 0, 'x': 0} False
        Q {'moment': 1, 'x': 1} True
        Q {'moment': 2, 'x': 2} False
        A {'moment': 0, 'x': 0} False
        Q {'moment': 3, 'x': 3} True
        A {'moment': 1, 'x': 1} True
        Q {'moment': 4, 'x': 4} False
        A {'moment': 2, 'x': 2} False
        A {'moment': 3, 'x': 3} True
        A {'moment': 4, 'x': 4} False

    """

    answers = []

    for x, y in X_y:

        while answers:

            # Get the oldest example
            x_ans, y_ans = answers[0]

            # If the oldest answer isn't old enough then stop
            if x_ans[on] + lag > x[on]:
                break

            # Else yield the oldest answer and release it from memory
            yield False, x_ans, y_ans
            del answers[0]

        # Show the observation and label it as a question
        yield True, x, y

        # Store the answer for the future
        answers.append((x, y))

    # Yield the final answers that remain
    for x, y in answers:
        yield False, x, y


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

    # Initialize the buffer
    buff = list(itertools.islice(stream, buffer_size))

    # Deplete the stream until it is empty
    for element in stream:

        # Pick a random element from the buffer and yield it
        i = rng.randint(0, len(buff) - 1)
        yield buff[i]

        # Replace the yielded element from the buffer with the new element from the stream
        buff[i] = element

    # Shuffle the remaining buffer elements and yield them one by one
    rng.shuffle(buff)
    for element in buff:
        yield element
