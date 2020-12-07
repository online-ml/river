import typing

import vaex
from vaex.utils import _ensure_strings_from_expressions, _ensure_list

from river import base


def iter_vaex(
    X: vaex.dataframe.DataFrame,
    y: typing.Union[str, vaex.expression.Expression] = None,
    features: typing.Union[typing.List[str], vaex.expression.Expression] = None,
) -> base.typing.Stream:
    """Yields rows from a ``vaex.DataFrame``.

    Parameters
    ----------
    X
        A vaex DataFrame housing the training featuers.
    y
        The column or expression containing the target variable.
    features
        A list of features used for training. If None, all columns in `X` will be used. Features
        specifying in `y` are ignored.

    """

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
                {key: X.evaluate(key, i, i + 1)[0] for key in y},
            )

    else:

        for i in range(len(X)):
            yield (
                {key: X.evaluate(key, i, i + 1)[0] for key in feature_names},
                X.evaluate(y[0], i, i + 1)[0],
            )
