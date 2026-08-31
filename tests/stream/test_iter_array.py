from __future__ import annotations

import operator
import typing

import numpy as np
import pytest

from river import stream

# Imported at runtime, rather than under TYPE_CHECKING, as `assert_type` evaluates its arguments.
from river.base.typing import FeatureName

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    Array = np.ndarray | Sequence[typing.Any]
    Rows = list[tuple[typing.Any, typing.Any]]

# The same data reaches `iter_array` as numpy or as plain Python containers, and both are
# documented as supported. Every case below is expected to yield the very same stream.
ARRAY_BACKENDS: dict[str, Callable[[list[typing.Any]], Array]] = {
    "numpy": np.array,
    "list": list,
    "tuple": tuple,
}

FEATURES = [[1, 2, 3], [11, 12, 13]]
LABELED = [{0: 1, 1: 2, 2: 3}, {0: 11, 1: 12, 2: 13}]
TARGET = [True, False]
MULTI_TARGET = [[1, 2], [11, 12]]
TEXTS = ["foo", "bar"]

# Enough rows for a shuffle to be all but certain to reorder them, each labeled with its own
# first feature so that a row and its target can be checked to travel together.
ROWS = [[i, i * 10] for i in range(10)]
LABELS = list(range(10))
MULTI_LABELS = [[i, -i] for i in range(10)]
LONG_TEXTS = [f"text {i}" for i in range(10)]
SHUFFLE: dict[str, typing.Any] = {"shuffle": True, "seed": 42}


@pytest.fixture(params=list(ARRAY_BACKENDS))
def array(request: pytest.FixtureRequest) -> Callable[[list[typing.Any]], Array]:
    """Yields one array-like constructor per supported input container."""
    return ARRAY_BACKENDS[request.param]


class Case(typing.NamedTuple):
    """One call to `iter_array`, with its arrays expressed as plain Python containers."""

    X: list[typing.Any]
    y: list[typing.Any] | None = None
    kwargs: dict[str, typing.Any] = {}


def run(case: Case, backend: Callable[[list[typing.Any]], Array]) -> Rows:
    return list(
        stream.iter_array(
            backend(case.X), None if case.y is None else backend(case.y), **case.kwargs
        )
    )


STREAMS: dict[str, tuple[Case, Rows]] = {
    "features-only": (Case(FEATURES), [(LABELED[0], None), (LABELED[1], None)]),
    "with-target": (Case(FEATURES, TARGET), [(LABELED[0], True), (LABELED[1], False)]),
    "named-features": (
        Case(FEATURES, TARGET, {"feature_names": ["x1", "x2", "x3"]}),
        [({"x1": 1, "x2": 2, "x3": 3}, True), ({"x1": 11, "x2": 12, "x3": 13}, False)],
    ),
    "fewer-names-than-features": (
        Case(FEATURES, kwargs={"feature_names": ["x1"]}),
        [({"x1": 1}, None), ({"x1": 11}, None)],
    ),
    "shorter-target-is-padded": (
        Case(FEATURES, [True]),
        [(LABELED[0], True), (LABELED[1], None)],
    ),
    "multioutput": (
        Case(FEATURES, MULTI_TARGET),
        [(LABELED[0], {0: 1, 1: 2}), (LABELED[1], {0: 11, 1: 12})],
    ),
    "named-outputs": (
        Case(FEATURES, MULTI_TARGET, {"target_names": ["y1", "y2"]}),
        [(LABELED[0], {"y1": 1, "y2": 2}), (LABELED[1], {"y1": 11, "y2": 12})],
    ),
    "target-names-ignored-for-a-single-output": (
        Case(FEATURES, TARGET, {"target_names": ["y1"]}),
        [(LABELED[0], True), (LABELED[1], False)],
    ),
    "text-passes-through": (Case(TEXTS, TARGET), [("foo", True), ("bar", False)]),
    "empty": (Case([], []), []),
    "empty-without-target": (Case([]), []),
}


@pytest.mark.parametrize(("case", "expected"), STREAMS.values(), ids=STREAMS)
def test_expected_stream(
    case: Case, expected: Rows, array: Callable[[list[typing.Any]], Array]
) -> None:
    """Each input shape yields its expected rows, whichever container the arrays come in.

    Features are labeled with their position when no names are given, and features without a
    name are dropped. When `y` is omitted, or shorter than `X`, the target is padded with `None`.
    A 2D target yields one dict per row, and a 1D array of texts is yielded as-is.
    """
    assert run(case, array) == expected


SHUFFLED_STREAMS: dict[str, Case] = {
    "with-target": Case(ROWS, LABELS, SHUFFLE),
    "multioutput": Case(ROWS, MULTI_LABELS, SHUFFLE),
    "without-target": Case(ROWS, None, SHUFFLE),
    "text": Case(LONG_TEXTS, LABELS, SHUFFLE),
}


@pytest.mark.parametrize("case", SHUFFLED_STREAMS.values(), ids=SHUFFLED_STREAMS)
def test_backends_agree_on_shuffling(case: Case) -> None:
    """numpy, lists and tuples are all reordered the same way for a given seed."""
    streams = [run(case, backend) for backend in ARRAY_BACKENDS.values()]
    assert all(rows == streams[0] for rows in streams)


def test_shuffle_reorders_and_preserves_rows(array: Callable[[list[typing.Any]], Array]) -> None:
    """Shuffling only reorders the rows, each one keeping its own target."""
    X, y = array(ROWS), array(LABELS)
    plain = list(stream.iter_array(X, y))
    shuffled = list(stream.iter_array(X, y, shuffle=True, seed=42))

    assert shuffled != plain
    assert all(xi[0] == yi for xi, yi in shuffled)
    assert sorted(shuffled, key=operator.itemgetter(1)) == plain


def test_shuffle_is_seeded(array: Callable[[list[typing.Any]], Array]) -> None:
    """The same seed always gives the same order, and different seeds give different ones."""
    X, y = array(ROWS), array(LABELS)

    assert list(stream.iter_array(X, y, **SHUFFLE)) == list(stream.iter_array(X, y, **SHUFFLE))
    assert list(stream.iter_array(X, y, shuffle=True, seed=0)) != list(
        stream.iter_array(X, y, shuffle=True, seed=1)
    )


def test_native_python_scalars(array: Callable[[list[typing.Any]], Array]) -> None:
    """Cells and targets are plain Python values, not numpy scalars."""
    xi, yi = next(iter(stream.iter_array(array(FEATURES), array(TARGET))))
    _, multi = next(iter(stream.iter_array(array(FEATURES), array(MULTI_TARGET))))
    text, _ = next(iter(stream.iter_array(array(TEXTS), array(TARGET))))

    assert [type(value) for value in xi.values()] == [int, int, int]
    assert type(yi) is bool
    assert [type(value) for value in multi.values()] == [int, int]
    assert isinstance(text, str)


def test_static_types_follow_the_input_shapes() -> None:
    """What a checker makes of each input shape, which `assert_type` fails the mypy run over."""
    # A sequence of texts yields texts, and a sequence of targets yields the target's own type.
    typing.assert_type(next(iter(stream.iter_array(["a", "b"], [1, 2]))), tuple[str, int])
    typing.assert_type(
        next(iter(stream.iter_array([[1, 2]], [True]))),
        tuple[dict[FeatureName, typing.Any], bool],
    )
    # numpy hands out `Any`, and so does a target a checker cannot tell apart from a 2D one.
    typing.assert_type(
        next(iter(stream.iter_array(np.array([[1, 2]]), np.array([1])))),
        tuple[dict[FeatureName, typing.Any], typing.Any],
    )
    typing.assert_type(
        next(iter(stream.iter_array([[1, 2]], [[1, 2]]))),
        tuple[dict[FeatureName, typing.Any], typing.Any],
    )


def test_iteration_is_lazy() -> None:
    """Nothing is read from the arrays until the stream is iterated over."""
    dataset = stream.iter_array(np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        _ = next(iter(dataset))
