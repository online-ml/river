from __future__ import annotations

import typing

import pytest

from river import stream
from tests.frames import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from river.base.typing import FeatureName
    from tests.frames import FrameBackend

    Data = dict[str, list[typing.Any]]
    Row = tuple[dict[str, typing.Any], bool]

# Canonical example, expressed once and reused across every backend. The expected output is
# backend-independent and made of plain Python scalars (no numpy/arrow scalar leaks).
FEATURES: Data = {"x1": [1, 2, 3, 4], "x2": ["blue", "yellow", "yellow", "blue"]}
TARGET: list[bool] = [True, False, False, True]
EXPECTED: list[Row] = [
    ({"x1": 1, "x2": "blue"}, True),
    ({"x1": 2, "x2": "yellow"}, False),
    ({"x1": 3, "x2": "yellow"}, False),
    ({"x1": 4, "x2": "blue"}, True),
]


def test_features_and_target(frame_backend: FrameBackend) -> None:
    """Every supported backend yields the same rows from features + a target series."""
    assert (
        list(stream.iter_frame(frame_backend.frame(FEATURES), frame_backend.series(TARGET)))
        == EXPECTED
    )


def test_native_python_scalars(frame_backend: FrameBackend) -> None:
    """Cells are plain Python scalars, not numpy/arrow scalars."""
    xi, yi = next(
        iter(stream.iter_frame(frame_backend.frame(FEATURES), frame_backend.series(TARGET)))
    )
    assert type(xi["x1"]) is int
    assert type(xi["x2"]) is str
    assert type(yi) is bool


def test_no_target(frame_backend: FrameBackend) -> None:
    """When `y` is omitted the target is `None` for every row."""
    rows = stream.iter_frame(frame_backend.frame(FEATURES))
    assert [(xi, yi) for xi, yi in rows] == [(xi, None) for xi, _ in EXPECTED]


def test_multioutput_target(frame_backend: FrameBackend) -> None:
    """A dataframe target yields one dict per row with a column per output."""
    X = frame_backend.frame({"x1": [1, 2]})
    Y = frame_backend.frame({"y1": [True, False], "y2": [False, True]})
    assert list(stream.iter_frame(X, Y)) == [
        ({"x1": 1}, {"y1": True, "y2": False}),
        ({"x1": 2}, {"y1": False, "y2": True}),
    ]


def test_mixed_dtypes_are_not_coerced(frame_backend: FrameBackend) -> None:
    """Per-column dtypes are preserved: an int column next to a float column stays int.

    Going through a single numpy array (the previous implementation) would upcast the whole
    frame to a common dtype, turning ``1`` into ``1.0``.
    """
    xi, _ = next(iter(stream.iter_frame(frame_backend.frame({"i": [1, 2], "f": [1.5, 2.5]}))))
    assert xi == {"i": 1, "f": 1.5}
    assert type(xi["i"]) is int
    assert type(xi["f"]) is float


def test_shuffle_is_seeded_and_preserves_rows(frame_backend: FrameBackend) -> None:
    """Shuffling is reproducible for a given seed and only reorders the rows."""
    X, y = frame_backend.frame(FEATURES), frame_backend.series(TARGET)
    shuffled = list(stream.iter_frame(X, y, shuffle=True, seed=42))

    assert shuffled == list(stream.iter_frame(X, y, shuffle=True, seed=42))  # same seed, same order
    assert shuffled != EXPECTED  # seed=42 actually permutes these four rows
    assert sorted(shuffled, key=lambda row: row[0]["x1"]) == EXPECTED  # same rows, different order


def test_different_seeds_give_different_orders(frame_backend: FrameBackend) -> None:
    X, y = frame_backend.frame(FEATURES), frame_backend.series(TARGET)
    assert list(stream.iter_frame(X, y, shuffle=True, seed=0)) != list(
        stream.iter_frame(X, y, shuffle=True, seed=1)
    )


def test_backends_agree() -> None:
    """All installed backends produce byte-for-byte identical streams."""
    streams: list[list[tuple[dict[FeatureName, typing.Any], typing.Any]]] = []
    for backend_builder in FRAME_BACKENDS.values():
        frame_backend = backend_builder()
        streams.append(
            [*stream.iter_frame(frame_backend.frame(FEATURES), frame_backend.series(TARGET))]
        )
    if not streams:
        pytest.skip("no dataframe backend installed")
    assert all(s == streams[0] for s in streams)


def test_lazy_frame_raises() -> None:
    """Lazy frames are rejected with an explicit, River-specific error."""
    pl = pytest.importorskip("polars")
    with pytest.raises(TypeError, match="only supports eager dataframes"):
        # iter_frame is a generator, so the error surfaces on first iteration.
        next(iter(stream.iter_frame(pl.LazyFrame(FEATURES))))


@pytest.mark.parametrize(
    ("iter_fn", "make_backend"),
    [
        (stream.iter_pandas, FRAME_BACKENDS["pandas"]),
        (stream.iter_polars, FRAME_BACKENDS["polars"]),
    ],
)
def test_deprecated_wrapper_forwards_shuffle(
    iter_fn: Callable[..., typing.Iterator[Row]], make_backend: Callable[[], FrameBackend]
) -> None:
    """The deprecated wrappers forward `shuffle`/`seed` through to `iter_frame`."""
    b = make_backend()
    with pytest.warns(DeprecationWarning):
        rows = list(iter_fn(b.frame(FEATURES), b.series(TARGET), shuffle=True, seed=42))
    assert sorted(rows, key=lambda row: row[0]["x1"]) == EXPECTED


@pytest.mark.parametrize(("n_x", "n_y"), [(3, 2), (2, 3)])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("multioutput", [False, True])
def test_mismatched_feature_target_lengths(
    frame_backend: FrameBackend,
    n_x: int,
    n_y: int,
    shuffle: bool,
    multioutput: bool,
) -> None:
    X = frame_backend.frame({"x": list(range(n_x))})

    y: typing.Any
    if multioutput:
        y = frame_backend.frame(
            {
                "y1": list(range(n_y)),
                "y2": list(range(n_y)),
            }
        )
    else:
        y = frame_backend.series(list(range(n_y)))

    with pytest.raises(ValueError, match="X and y must have the same length"):
        list(stream.iter_frame(X, y, shuffle=shuffle, seed=42))


@pytest.mark.parametrize(
    ("dtype", "values", "scalar_type"),
    [
        ("Int64", [2**53 + 1, -(2**53) - 1], int),
        ("UInt64", [2**63 + 1, 2**63 + 3], int),
        ("Float64", [1.5, -2.5], float),
        ("boolean", [True, False], bool),
    ],
)
@pytest.mark.parametrize("target_kind", ["none", "series", "frame"])
def test_pandas_nullable_native_scalars(
    dtype: str, values: list[typing.Any], scalar_type: type[object], target_kind: str
) -> None:
    pd = pytest.importorskip("pandas")
    expected_values = [values[0], None, values[1]]
    X = pd.DataFrame({"x": pd.Series(expected_values, dtype=dtype)})
    if target_kind == "none":
        y = None
    elif target_kind == "series":
        y = X["x"]
    else:
        y = X.rename(columns={"x": "y"})

    rows = list(stream.iter_frame(X, y))

    for (features, target), expected in zip(rows, expected_values, strict=True):
        if expected is None:
            assert features["x"] is None
        else:
            assert type(features["x"]) is scalar_type
            assert features["x"] == expected
        if target_kind == "frame":
            target = target["y"]
        if target_kind == "none" or expected is None:
            assert target is None
        else:
            assert type(target) is scalar_type
            assert target == expected


def test_pandas_nullable_rows_across_batches() -> None:
    pd = pytest.importorskip("pandas")
    values: list[int | None] = [2**63 + i for i in range(1025)]
    values[512] = None
    X = pd.DataFrame({"x": pd.Series(values, dtype="UInt64")})

    rows = list(stream.iter_frame(X, X["x"], shuffle=True, seed=7))
    repeated = list(stream.iter_frame(X, X["x"], shuffle=True, seed=7))

    assert rows == repeated
    assert len(rows) == len(values)
    assert {features["x"] for features, _ in rows} == set(values)
    assert all(features["x"] == target for features, target in rows)
    assert all(type(target) is int or target is None for _, target in rows)


@pytest.mark.parametrize("name", [None, "", 0, "target"])
def test_pandas_nullable_series_target_names(name: typing.Any) -> None:
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([2**53 + 1, None], dtype="Int64", name=name)

    rows = list(stream.iter_frame(X, y))

    assert rows == [({"x": 1}, 2**53 + 1), ({"x": 2}, None)]
    assert type(rows[0][1]) is int
