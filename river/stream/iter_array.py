from __future__ import annotations

import operator
import random
import typing
from collections.abc import Mapping, Sequence, Sized

import numpy as np

from river import base

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection

    Features: typing.TypeAlias = dict[base.typing.FeatureName, typing.Any]
    Row: typing.TypeAlias = str | Collection[object]

_T = typing.TypeVar("_T")
_RowT = typing.TypeVar("_RowT", bound="Row")


@typing.overload
def _take(array: np.ndarray, order: list[int]) -> np.ndarray: ...


@typing.overload
def _take(array: list[_T], order: list[int]) -> list[_T]: ...


def _take(array: np.ndarray | list[_T], order: list[int]) -> np.ndarray | list[_T]:
    return array[order] if isinstance(array, np.ndarray) else [array[i] for i in order]


def _is_feature_row(row: object) -> typing.TypeGuard[Sized]:
    return isinstance(row, Sized) and not isinstance(row, Mapping)


def _passthrough(row: Row) -> Row:
    return row.tolist() if isinstance(row, np.ndarray) else row


def _labeler(names: Sequence[base.typing.FeatureName]) -> Callable[[Row], Features]:
    def label(row: Row) -> Features:
        return dict(zip(names, row.tolist() if isinstance(row, np.ndarray) else row))

    return label


def iter_array(
    X: np.ndarray | list[_RowT],
    y: np.ndarray | list[typing.Any] | None = None,
    feature_names: Sequence[base.typing.FeatureName] | None = None,
    target_names: Sequence[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> base.typing.Stream:
    """Iterates over the rows from an array of features and an array of targets.

    This method is intended to work with `numpy` arrays, but should also work with Python lists.

    Parameters
    ----------
    X
        A 2D array of features. This can also be a 1D array of strings, which can be the case if
        you're working with text.
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

    This also works with an array of texts:

    >>> X = ["foo", "bar"]
    >>> dataset = stream.iter_array(X, Y)
    >>> for x, y in dataset:
    ...     print(x, y)
    foo True
    bar False

    """

    n_rows = len(X)
    if y is not None and len(y) != n_rows:
        raise ValueError(f"X and y must have the same length, got {n_rows} and {len(y)}")

    if n_rows == 0:
        return

    first_row = X[0]
    to_x: Callable[[Row], Features]
    if isinstance(first_row, str):
        to_x = typing.cast("Callable[[Row], Features]", _passthrough)

    elif not _is_feature_row(first_row):
        raise ValueError(
            "X must be a 2D array or a 1D array of strings, got rows of type "
            f"{type(first_row).__name__}; use X.reshape(-1, 1) for a single feature"
        )

    else:
        to_x = _labeler(list(range(len(first_row))) if feature_names is None else feature_names)

    output_names: Sequence[base.typing.FeatureName] | None = None
    if y is not None and np.ndim(y[0]) > 0:
        output_names = list(range(len(y[0]))) if target_names is None else target_names

    rng = random.Random(seed)
    order = rng.sample(range(n_rows), k=n_rows) if shuffle else None
    rows = X if order is None else _take(X, order)
    if isinstance(rows, np.ndarray) and rows.dtype.kind == "U":
        rows = rows.tolist()

    if y is None:
        for row in rows:
            yield to_x(row), None
        return

    targets = y if order is None else _take(y, order)
    if output_names is None:
        if isinstance(targets, np.ndarray):
            targets = targets.tolist()
        for row, target in zip(rows, targets):
            yield to_x(row), target

    else:
        tolist = operator.methodcaller("tolist")
        target_rows = map(tolist, targets) if isinstance(y[0], np.ndarray) else targets
        for row, outputs in zip(rows, target_rows):
            yield to_x(row), dict(zip(output_names, outputs))
