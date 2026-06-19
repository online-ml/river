from __future__ import annotations

import typing

import pytest

from river import stream
from river.conftest import BACKENDS, Backend

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from river.base.typing import FeatureName

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


def test_features_and_target(backend: Backend) -> None:
    """Every supported backend yields the same rows from features + a target series."""
    assert list(stream.iter_frame(backend.frame(FEATURES), backend.series(TARGET))) == EXPECTED


def test_native_python_scalars(backend: Backend) -> None:
    """Cells are plain Python scalars, not numpy/arrow scalars."""
    xi, yi = next(iter(stream.iter_frame(backend.frame(FEATURES), backend.series(TARGET))))
    assert type(xi["x1"]) is int
    assert type(xi["x2"]) is str
    assert type(yi) is bool


def test_no_target(backend: Backend) -> None:
    """When `y` is omitted the target is `None` for every row."""
    rows = stream.iter_frame(backend.frame(FEATURES))
    assert [(xi, yi) for xi, yi in rows] == [(xi, None) for xi, _ in EXPECTED]


def test_multioutput_target(backend: Backend) -> None:
    """A dataframe target yields one dict per row with a column per output."""
    X = backend.frame({"x1": [1, 2]})
    Y = backend.frame({"y1": [True, False], "y2": [False, True]})
    assert list(stream.iter_frame(X, Y)) == [
        ({"x1": 1}, {"y1": True, "y2": False}),
        ({"x1": 2}, {"y1": False, "y2": True}),
    ]


def test_mixed_dtypes_are_not_coerced(backend: Backend) -> None:
    """Per-column dtypes are preserved: an int column next to a float column stays int.

    Going through a single numpy array (the previous implementation) would upcast the whole
    frame to a common dtype, turning ``1`` into ``1.0``.
    """
    xi, _ = next(iter(stream.iter_frame(backend.frame({"i": [1, 2], "f": [1.5, 2.5]}))))
    assert xi == {"i": 1, "f": 1.5}
    assert type(xi["i"]) is int
    assert type(xi["f"]) is float


def test_shuffle_is_seeded_and_preserves_rows(backend: Backend) -> None:
    """Shuffling is reproducible for a given seed and only reorders the rows."""
    X, y = backend.frame(FEATURES), backend.series(TARGET)
    shuffled = list(stream.iter_frame(X, y, shuffle=True, seed=42))

    assert shuffled == list(stream.iter_frame(X, y, shuffle=True, seed=42))  # same seed, same order
    assert shuffled != EXPECTED  # seed=42 actually permutes these four rows
    assert sorted(shuffled, key=lambda row: row[0]["x1"]) == EXPECTED  # same rows, different order


def test_different_seeds_give_different_orders(backend: Backend) -> None:
    X, y = backend.frame(FEATURES), backend.series(TARGET)
    assert list(stream.iter_frame(X, y, shuffle=True, seed=0)) != list(
        stream.iter_frame(X, y, shuffle=True, seed=1)
    )


def test_backends_agree() -> None:
    """All installed backends produce byte-for-byte identical streams."""
    streams: list[list[tuple[dict[FeatureName, typing.Any], typing.Any]]] = []
    for backend_builder in BACKENDS.values():
        backend = backend_builder()
        streams.append([*stream.iter_frame(backend.frame(FEATURES), backend.series(TARGET))])
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
    [(stream.iter_pandas, BACKENDS["pandas"]), (stream.iter_polars, BACKENDS["polars"])],
)
def test_deprecated_wrapper_forwards_shuffle(
    iter_fn: Callable[..., typing.Iterator[Row]], make_backend: Callable[[], Backend]
) -> None:
    """The deprecated wrappers forward `shuffle`/`seed` through to `iter_frame`."""
    b = make_backend()
    with pytest.warns(DeprecationWarning):
        rows = list(iter_fn(b.frame(FEATURES), b.series(TARGET), shuffle=True, seed=42))
    assert sorted(rows, key=lambda row: row[0]["x1"]) == EXPECTED
