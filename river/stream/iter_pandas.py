import typing

import pandas as pd

from river import base, stream


def iter_pandas(
    X: pd.DataFrame, y: typing.Union[pd.Series, pd.DataFrame] = None, **kwargs
) -> base.typing.Stream:
    """Iterates over the rows of a `pandas.DataFrame`.

    Parameters
    ----------
    X
        A dataframe of features.
    y
        A series or a dataframe with one column per target.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.

    Examples
    --------

    >>> import pandas as pd
    >>> from river import stream

    >>> X = pd.DataFrame({
    ...     'x1': [1, 2, 3, 4],
    ...     'x2': ['blue', 'yellow', 'yellow', 'blue'],
    ...     'y': [True, False, False, True]
    ... })
    >>> y = X.pop('y')

    >>> for xi, yi in stream.iter_pandas(X, y):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    {'x1': 4, 'x2': 'blue'} True

    """

    kwargs["feature_names"] = X.columns
    if isinstance(y, pd.DataFrame):
        kwargs["target_names"] = y.columns

    yield from stream.iter_array(
        X=X.to_numpy(), y=y if y is None else y.to_numpy(), **kwargs
    )
