from __future__ import annotations

import enum
import typing

import numpy as np
import pytest

from river import stream

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from river.base.typing import FeatureName

    Backend = typing.Literal["numpy", "list"]
    Rows = list[tuple[typing.Any, typing.Any]]

    class IterArrayOptions(typing.TypedDict, total=False):
        feature_names: list[FeatureName]
        target_names: list[FeatureName]
        shuffle: bool
        seed: int


ISSUE = "https://github.com/online-ml/river/issues/2046"
DID_NOT_RAISE = pytest.fail.Exception

KNOWN_BUGS: dict[tuple[str, ...], type[BaseException]] = {
    ("test_expected_stream", "numpy", "empty"): IndexError,
    ("test_expected_stream", "numpy", "empty-without-target"): IndexError,
    ("test_expected_stream", "list", "empty"): IndexError,
    ("test_expected_stream", "list", "empty-without-target"): IndexError,
    ("test_invalid_input_raises", "numpy", "shorter-target"): DID_NOT_RAISE,
    ("test_invalid_input_raises", "numpy", "shorter-multioutput-target"): TypeError,
    ("test_invalid_input_raises", "numpy", "longer-target"): TypeError,
    ("test_invalid_input_raises", "numpy", "longer-target-for-texts"): DID_NOT_RAISE,
    ("test_invalid_input_raises", "numpy", "empty-features-with-target"): IndexError,
    ("test_invalid_input_raises", "numpy", "1d-features"): TypeError,
    ("test_invalid_input_raises", "numpy", "rows-of-dicts"): DID_NOT_RAISE,
    ("test_invalid_input_raises", "list", "shorter-target"): DID_NOT_RAISE,
    ("test_invalid_input_raises", "list", "shorter-multioutput-target"): TypeError,
    ("test_invalid_input_raises", "list", "longer-target"): TypeError,
    ("test_invalid_input_raises", "list", "longer-target-for-texts"): DID_NOT_RAISE,
    ("test_invalid_input_raises", "list", "empty-features-with-target"): IndexError,
    ("test_invalid_input_raises", "list", "1d-features"): TypeError,
    ("test_invalid_input_raises", "list", "rows-of-dicts"): DID_NOT_RAISE,
}

ARRAY_BACKENDS: dict[Backend, Callable[[list[typing.Any]], typing.Any]] = {
    "numpy": np.asarray,
    "list": list,
}

FEATURES = [[1, 2, 3], [11, 12, 13]]
LABELED_FEATURES = [{0: 1, 1: 2, 2: 3}, {0: 11, 1: 12, 2: 13}]
TARGET = [True, False]
MULTI_TARGET = [[1, 2], [11, 12]]
MULTI_TARGET_ROWS: list[typing.Any] = list(np.asarray(MULTI_TARGET))
TEXTS = ["foo", "bar"]

LONG_FEATURES = [[i, i * 10] for i in range(10)]
LONG_TARGET = list(range(10))
LONG_MULTI_TARGET = [[i, -i] for i in range(10)]
LONG_TEXTS = [f"text {i}" for i in range(10)]
SEEDED_SHUFFLE: IterArrayOptions = {"shuffle": True, "seed": 42}


class Label(enum.StrEnum):
    A = "a"
    B = "b"


class Case(typing.NamedTuple):
    X: list[typing.Any]
    y: list[typing.Any] | None = None
    options: IterArrayOptions = {}


class ReadError(Exception):
    pass


class Unreadable:
    def __len__(self) -> int:
        raise ReadError

    def __getitem__(self, index: object) -> typing.NoReturn:
        raise ReadError


STREAMS: dict[str, tuple[Case, Rows]] = {
    "features-only": (
        Case(FEATURES),
        [(LABELED_FEATURES[0], None), (LABELED_FEATURES[1], None)],
    ),
    "with-target": (
        Case(FEATURES, TARGET),
        [(LABELED_FEATURES[0], True), (LABELED_FEATURES[1], False)],
    ),
    "named-features": (
        Case(FEATURES, TARGET, {"feature_names": ["x1", "x2", "x3"]}),
        [({"x1": 1, "x2": 2, "x3": 3}, True), ({"x1": 11, "x2": 12, "x3": 13}, False)],
    ),
    "fewer-names-than-features": (
        Case(FEATURES, options={"feature_names": ["x1"]}),
        [({"x1": 1}, None), ({"x1": 11}, None)],
    ),
    "more-names-than-features": (
        Case(FEATURES, options={"feature_names": ["x1", "x2", "x3", "x4"]}),
        [({"x1": 1, "x2": 2, "x3": 3}, None), ({"x1": 11, "x2": 12, "x3": 13}, None)],
    ),
    "multioutput": (
        Case(FEATURES, MULTI_TARGET),
        [(LABELED_FEATURES[0], {0: 1, 1: 2}), (LABELED_FEATURES[1], {0: 11, 1: 12})],
    ),
    "multioutput-array-rows": (
        Case(FEATURES, MULTI_TARGET_ROWS),
        [(LABELED_FEATURES[0], {0: 1, 1: 2}), (LABELED_FEATURES[1], {0: 11, 1: 12})],
    ),
    "named-outputs": (
        Case(FEATURES, MULTI_TARGET, {"target_names": ["y1", "y2"]}),
        [
            (LABELED_FEATURES[0], {"y1": 1, "y2": 2}),
            (LABELED_FEATURES[1], {"y1": 11, "y2": 12}),
        ],
    ),
    "target-names-ignored-for-a-single-output": (
        Case(FEATURES, TARGET, {"target_names": ["y1"]}),
        [(LABELED_FEATURES[0], True), (LABELED_FEATURES[1], False)],
    ),
    "str-enum-labels": (
        Case(FEATURES, [Label.A, Label.B]),
        [(LABELED_FEATURES[0], Label.A), (LABELED_FEATURES[1], Label.B)],
    ),
    "none-as-first-label": (
        Case(FEATURES, [None, True]),
        [(LABELED_FEATURES[0], None), (LABELED_FEATURES[1], True)],
    ),
    "dict-labels": (
        Case(FEATURES, [{"y": 1}, {"y": 2}]),
        [(LABELED_FEATURES[0], {"y": 1}), (LABELED_FEATURES[1], {"y": 2})],
    ),
    "text-passes-through": (Case(TEXTS, TARGET), [("foo", True), ("bar", False)]),
    "text-features": (
        Case([["a", "b"], ["c", "d"]]),
        [({0: "a", 1: "b"}, None), ({0: "c", 1: "d"}, None)],
    ),
    "empty": (Case([], []), []),
    "empty-without-target": (Case([]), []),
}

INVALID_STREAMS: dict[str, Case] = {
    "shorter-target": Case(FEATURES, [True]),
    "shorter-multioutput-target": Case(FEATURES, [[1, 2]]),
    "longer-target": Case([[1, 2]], [True, False]),
    "longer-target-for-texts": Case(["foo"], [True, False]),
    "empty-features-with-target": Case([], [True]),
    "1d-features": Case([1.0, 2.0]),
    "rows-of-dicts": Case([{"a": 1}, {"b": 2}]),
}

SHUFFLED_STREAMS: dict[str, Case] = {
    "with-target": Case(LONG_FEATURES, LONG_TARGET, SEEDED_SHUFFLE),
    "multioutput": Case(LONG_FEATURES, LONG_MULTI_TARGET, SEEDED_SHUFFLE),
    "without-target": Case(LONG_FEATURES, None, SEEDED_SHUFFLE),
    "text": Case(LONG_TEXTS, LONG_TARGET, SEEDED_SHUFFLE),
}


def xfail_if_known_bug(request: pytest.FixtureRequest, key: tuple[str, ...]) -> None:
    if (raises := KNOWN_BUGS.get(key)) is not None:
        request.applymarker(pytest.mark.xfail(raises=raises, strict=True, reason=ISSUE))


def collect_rows(case: Case, backend: Backend) -> Rows:
    array = ARRAY_BACKENDS[backend]
    y = None if case.y is None else array(case.y)
    return list(stream.iter_array(array(case.X), y, **case.options))


def cells(rows: Rows) -> Iterator[object]:
    for row in rows:
        for part in row:
            yield from part.values() if isinstance(part, dict) else (part,)


@pytest.fixture(params=list(ARRAY_BACKENDS))
def backend(request: pytest.FixtureRequest) -> Backend:
    return typing.cast("Backend", request.param)


@pytest.mark.parametrize("name", STREAMS)
def test_expected_stream(name: str, backend: Backend, request: pytest.FixtureRequest) -> None:
    xfail_if_known_bug(request, ("test_expected_stream", backend, name))
    case, expected = STREAMS[name]
    rows = collect_rows(case, backend)
    assert rows == expected
    assert not [cell for cell in cells(rows) if isinstance(cell, np.generic)]


@pytest.mark.parametrize("name", INVALID_STREAMS)
def test_invalid_input_raises(name: str, backend: Backend, request: pytest.FixtureRequest) -> None:
    xfail_if_known_bug(request, ("test_invalid_input_raises", backend, name))
    with pytest.raises(ValueError):
        _ = collect_rows(INVALID_STREAMS[name], backend)


@pytest.mark.parametrize("name", SHUFFLED_STREAMS)
def test_backends_agree_on_shuffling(name: str) -> None:
    case = SHUFFLED_STREAMS[name]
    assert collect_rows(case, "numpy") == collect_rows(case, "list")


@pytest.mark.parametrize("name", SHUFFLED_STREAMS)
def test_shuffle_reorders_and_preserves_rows(name: str, backend: Backend) -> None:
    case = SHUFFLED_STREAMS[name]
    shuffled = collect_rows(case, backend)
    plain = collect_rows(case._replace(options={}), backend)

    assert shuffled != plain
    assert sorted(shuffled, key=plain.index) == plain
    assert not [cell for cell in cells(shuffled) if isinstance(cell, np.generic)]


def test_list_targets_shuffle_like_numpy_targets() -> None:
    X = np.array(LONG_FEATURES)
    y: typing.Any = LONG_TARGET
    assert list(stream.iter_array(X, y, **SEEDED_SHUFFLE)) == list(
        stream.iter_array(X, np.array(LONG_TARGET), **SEEDED_SHUFFLE)
    )


def test_shuffle_is_seeded(backend: Backend) -> None:
    case = Case(LONG_FEATURES, LONG_TARGET, SEEDED_SHUFFLE)

    assert collect_rows(case, backend) == collect_rows(case, backend)
    assert collect_rows(case._replace(options={"shuffle": True, "seed": 0}), backend) != (
        collect_rows(case._replace(options={"shuffle": True, "seed": 1}), backend)
    )


@pytest.mark.parametrize("y", [None, Unreadable()], ids=["without-target", "with-target"])
def test_iteration_is_lazy(y: typing.Any) -> None:
    X: typing.Any = Unreadable()
    dataset = stream.iter_array(X, y)
    with pytest.raises(ReadError):
        _ = next(dataset)
