from __future__ import annotations

import itertools
import random
import typing

import numpy as np

from river import base

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

Target = typing.TypeVar("Target", bound=base.typing.Target)
"""The type of a single target value, i.e. of a row of a 1D array of targets."""

Array: typing.TypeAlias = "np.ndarray | Sequence[typing.Any]"
"""A numpy array or a plain Python sequence."""

Features: typing.TypeAlias = "dict[base.typing.FeatureName, typing.Any]"
"""One row of features, labeled, which is what an estimator takes as its `x`."""


def _passthrough(row: str) -> str:
    """Leaves a row as it is, which is how an array of texts is yielded."""
    return row


def _labeler(
    names: Sequence[base.typing.FeatureName], first_row: typing.Any
) -> Callable[[typing.Any], Features]:
    """Picks the labeler for every row from the first one, all rows being of the same type."""
    if isinstance(first_row, np.ndarray):

        def label_numpy(row: np.ndarray) -> Features:
            return dict(zip(names, row.tolist()))

        return label_numpy

    def label(row: typing.Any) -> Features:
        return dict(zip(names, row))

    return label


def _take(values: Array, order: Sequence[int]) -> Array:
    """Reorders rows, with fancy indexing for numpy and row lookups for plain sequences."""
    return values[order] if isinstance(values, np.ndarray) else [values[i] for i in order]


# NOTE: A row of X is labeled to make a dictionary of features, unless X is an array of texts,
# which are yielded as they are. A row of y is a target, unless y is multi-output, in which case
# it is a dictionary too. A plain Python sequence says which of these it is, and the overloads
# below follow it; a numpy array hands out `Any`, which is where the trail stops. The first
# overload overlaps the wider ones by design, which is what the pyright suppression is for.
@typing.overload
def iter_array(  # pyright: ignore[reportOverlappingOverload]
    X: Sequence[str],
    y: Sequence[Target],
    feature_names: list[base.typing.FeatureName] | None = None,
    target_names: list[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[str, Target]]: ...


@typing.overload
def iter_array(
    X: Sequence[str],
    y: Array | None = None,
    feature_names: list[base.typing.FeatureName] | None = None,
    target_names: list[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[str, typing.Any]]: ...


@typing.overload
def iter_array(
    X: Array,
    y: Sequence[Target],
    feature_names: list[base.typing.FeatureName] | None = None,
    target_names: list[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[Features, Target]]: ...


@typing.overload
def iter_array(
    X: Array,
    y: Array | None = None,
    feature_names: list[base.typing.FeatureName] | None = None,
    target_names: list[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> base.typing.Stream: ...


def iter_array(
    X: Array,
    y: Array | None = None,
    feature_names: list[base.typing.FeatureName] | None = None,
    target_names: list[base.typing.FeatureName] | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[Features | str, typing.Any]]:
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

    This also works with a array of texts:

    >>> X = ["foo", "bar"]
    >>> dataset = stream.iter_array(
    ...     X, Y,
    ...     feature_names=['x1', 'x2', 'x3']
    ... )
    >>> for x, y in dataset:
    ...     print(x, y)
    foo True
    bar False

    """
    if (n_rows := len(X)) == 0:
        return

    if shuffle:
        order = random.Random(seed).sample(range(n_rows), k=n_rows)
        X = _take(X, order)
        y = y if y is None else _take(y, order)

    handle_features: Callable[[typing.Any], Features | str]
    # If the first row of X is actually a string, then we assume all the rows are strings and will
    # pass them through. If not we assume each row is a set of features, and will label them.
    if isinstance(X[0], str):
        handle_features = _passthrough
    else:
        handle_features = _labeler(
            range(len(X[0])) if feature_names is None else feature_names, X[0]
        )

    # zip_longest pads a target array shorter than X with Nones, instead of stopping short of it.
    rows = itertools.zip_longest(X, () if y is None else y)

    if y is not None and not np.isscalar(y[0]):
        handle_target = _labeler(range(len(y[0])) if target_names is None else target_names, y[0])
        for xi, yi in rows:
            yield handle_features(xi), handle_target(yi)
    else:
        for xi, yi in rows:
            yield handle_features(xi), yi.item() if isinstance(yi, np.generic) else yi
