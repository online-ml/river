from __future__ import annotations

import itertools
import operator
import random
import typing
from collections.abc import Mapping, Sequence, Sized

import numpy as np

from river import base

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Iterator

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


def _tolist_chunks(
    array: np.ndarray, order: list[int] | None, chunk_size: int
) -> Iterator[typing.Any]:
    starts = range(0, len(array), chunk_size)
    if order is None:
        chunks = (array[i : i + chunk_size].tolist() for i in starts)
    else:
        chunks = (array[order[i : i + chunk_size]].tolist() for i in starts)
    return itertools.chain.from_iterable(chunks)


def _is_feature_row(row: object) -> typing.TypeGuard[Sized]:
    return isinstance(row, Sized) and not isinstance(row, Mapping)


def _passthrough(row: Row) -> Row:
    return row.tolist() if isinstance(row, np.ndarray) else row


def _labeler(
    names: Sequence[base.typing.FeatureName],
) -> Callable[[Iterable[Iterable[object]]], Iterator[Features]]:
    def label(rows: Iterable[Iterable[object]]) -> Iterator[Features]:
        return map(dict, map(zip, itertools.repeat(names), rows))

    return label


def iter_array(
    X: np.ndarray | list[_RowT],
    y: np.ndarray | list[typing.Any] | None = None,
    feature_names: Sequence[base.typing.FeatureName] | None = None,
    target_names: Sequence[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    chunk_size: int = 1024,
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

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

    n_rows = len(X)
    if y is not None and len(y) != n_rows:
        raise ValueError(f"X and y must have the same length, got {n_rows} and {len(y)}")

    if n_rows == 0:
        return

    first_row = X[0]
    to_x: Callable[[Iterable[Row]], Iterable[Features]]
    if isinstance(first_row, str):
        to_x = typing.cast("Callable[[Iterable[Row]], Iterable[Features]]", iter)

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
    tolist = operator.methodcaller("tolist")

    rows: Iterable[Row]
    if isinstance(X, np.ndarray) and X.ndim > 1:
        rows = map(tolist, X if order is None else map(X.__getitem__, order))
    elif isinstance(X, np.ndarray) and X.dtype.kind == "U":
        rows = _tolist_chunks(X, order, chunk_size)
    else:
        ordered = X if order is None else _take(X, order)
        has_array_rows = any(map(isinstance, X, itertools.repeat(np.ndarray)))
        rows = map(_passthrough, ordered) if has_array_rows else ordered

    if y is None:
        yield from zip(to_x(rows), itertools.repeat(None))
        return

    targets: Iterable[typing.Any]
    if output_names is None and isinstance(y, np.ndarray):
        targets = _tolist_chunks(y, order, chunk_size)
    else:
        targets = y if order is None else _take(y, order)

    if output_names is None:
        yield from zip(to_x(rows), targets)

    else:
        target_rows = map(tolist, targets) if isinstance(y[0], np.ndarray) else targets
        yield from zip(to_x(rows), _labeler(output_names)(target_rows))
