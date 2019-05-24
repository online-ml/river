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
    'iter_sklearn_dataset',
    'simulate_qa'
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
        yield dict(zip(feature_names, x)), yi


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
    for x, yi in iter_numpy(X.to_numpy(), y, **kwargs):
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


def simulate_qa(X_y, on, lag):
    """Simulate a real case learning scenario using a temporal attribute and a lag.

    Each observation will first be shown without revealing the ``y`` value. Once the observation is
    old enough it will be shown once again, this time with the ``y`` value being revealed. Each
    observation will thus be shown twice, once without the target equal to ``None`` and once with
    the actual value. The duration between "questions" and "answers" depends on the ``lag``
    parameter. For this to work the data is assumed to be sorted with respect to the temporal
    attribute.

    Parameters:
        X_y (generator): A stream of (``x``, ``y``) pairs.
        on (str): The attribute used for measuring time.
        lag (datetime.timedelta or int or float): Amount to wait before revealing the target
            associated with each observation. This value is expected to be able to sum with the
            `on` attribute.

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
