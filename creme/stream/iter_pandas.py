import typing

import pandas as pd

from creme import base
from creme import stream


def iter_pandas(X: pd.DataFrame, y: typing.Union[pd.Series, pd.DataFrame] = None,
                **kwargs) -> base.typing.Stream:
    """Iterates over the of a ``pandas.DataFrame``.

    Parameters:
        X: A dataframe of features.
        y: A series or a dataframe with one column per target.

    """
    import pandas as pd

    kwargs['feature_names'] = X.columns
    if isinstance(y, pd.DataFrame):
        kwargs['target_names'] = y.columns

    yield from stream.iter_array(X.to_numpy(), y.to_numpy(), **kwargs)
