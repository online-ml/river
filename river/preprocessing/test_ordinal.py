from __future__ import annotations

import random
import string
import typing

import narwhals.stable.v2 as nw
import pytest

from river import preprocessing
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from typing import Any

    from narwhals.stable.v2.typing import IntoDataFrame

    from river.conftest import FrameBackend

# `OrdinalEncoder.learn_many`/`transform_many` are routed through narwhals so any eager backend
# works. `learn_many` assigns codes in first-appearance order (`unique(maintain_order=True)`),
# which makes the encoding reproducible across backends and consistent with `learn_one`. The
# parametrization below pins this for several `unknown_value`/`none_value` combinations, which
# drive both the reserved-code bookkeeping (`learn_many`) and the encoded output (`transform_many`).

ENCODER_PARAMS = (
    pytest.param({}, id="defaults"),
    pytest.param({"unknown_value": None}, id="unknown_none"),
    pytest.param({"unknown_value": 5, "none_value": 10}, id="reserved_5_10"),
    pytest.param({"unknown_value": -1, "none_value": -2}, id="both_negative"),
    pytest.param({"unknown_value": 1, "none_value": 2}, id="reserved_1_2"),
)


def _categorical_batch(n: int = 40) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Build a reproducible batch of two categorical columns as (columns, row dicts)."""
    rng = random.Random(42)
    alphabet = list(string.ascii_lowercase[:5])
    rows = [{"c1": rng.choice(alphabet), "c2": rng.choice(alphabet)} for _ in range(n)]
    data = {"c1": [r["c1"] for r in rows], "c2": [r["c2"] for r in rows]}
    return data, rows


def _unknowns_batch() -> tuple[
    dict[str, list[str | None]], dict[str, list[str | None]], list[dict[str, str | None]]
]:
    """Return (learn batch, transform batch, transform rows).

    The transform batch deliberately mixes learned categories, an unseen category, and a null
    per column, so encoding exercises code lookup, ``unknown_value``, and ``none_value`` together.
    """
    learn: dict[str, list[str | None]] = {
        "c1": ["a", "b", "a", "c", "b", "c"],
        "c2": ["x", "y", "z", "x", "y", "z"],
    }
    trans: dict[str, list[str | None]] = {
        "c1": ["a", "d", "b", None, "c", "e"],
        "c2": ["x", "w", None, "y", "z", "x"],
    }
    rows = [{col: values[i] for col, values in trans.items()} for i in range(len(trans["c1"]))]
    return learn, trans, rows


def _read_rows(native: IntoDataFrame) -> list[dict[str, int | None]]:
    """Read a native encoded frame into rows, mapping nulls to ``None`` and codes to ``int``.

    The null mask is read separately so this stays correct across backends (a pandas nullable
    ``Int64`` null is ``pd.NA``, not ``None``), which matters when ``unknown_value=None`` makes
    unknown categories encode to null.
    """
    frame = nw.from_native(native, eager_only=True)
    columns = {col: (frame[col].is_null(), frame[col]) for col in frame.columns}
    return [
        {col: (None if mask[i] else int(values[i])) for col, (mask, values) in columns.items()}
        for i in range(len(frame))
    ]


@pytest.mark.parametrize("params", list(ENCODER_PARAMS))
def test_learn_many_codes_match_across_backends(
    frame_backend: FrameBackend, params: dict[str, Any]
) -> None:
    """Category codes are assigned identically regardless of the input backend."""
    data, _ = _categorical_batch()

    reference = preprocessing.OrdinalEncoder(**params)
    reference.learn_many(FRAME_BACKENDS["pandas"]().frame(data))

    encoder = preprocessing.OrdinalEncoder(**params)
    encoder.learn_many(frame_backend.frame(data))

    assert encoder.values == reference.values


@pytest.mark.parametrize("params", list(ENCODER_PARAMS))
def test_learn_many_matches_learn_one(frame_backend: FrameBackend, params: dict[str, Any]) -> None:
    """`learn_many` assigns the same codes as row-by-row `learn_one`."""
    data, rows = _categorical_batch()

    one = preprocessing.OrdinalEncoder(**params)
    for row in rows:
        one.learn_one(row)

    many = preprocessing.OrdinalEncoder(**params)
    many.learn_many(frame_backend.frame(data))

    assert many.values == one.values


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, {"a": 1, "b": 2, "c": 3, "d": 4}),
        ({"unknown_value": None}, {"a": 0, "b": 1, "c": 2, "d": 3}),
        ({"unknown_value": 1, "none_value": 2}, {"a": 0, "b": 3, "c": 4, "d": 5}),
        ({"unknown_value": 5, "none_value": 10}, {"a": 0, "b": 1, "c": 2, "d": 3}),
    ],
)
def test_learn_many_skips_reserved_codes(
    frame_backend: FrameBackend, params: dict[str, Any], expected: dict[str, int]
) -> None:
    """Non-negative `unknown_value`/`none_value` are reserved and never assigned to a category."""
    encoder = preprocessing.OrdinalEncoder(**params)
    encoder.learn_many(frame_backend.frame({"c": ["a", "b", "c", "d"]}))

    assert dict(encoder.values["c"]) == expected


@pytest.mark.parametrize("params", list(ENCODER_PARAMS))
def test_transform_many_is_backend_agnostic(
    frame_backend: FrameBackend, params: dict[str, Any]
) -> None:
    """`transform_many` yields identical codes regardless of the input backend."""
    learn, trans, _ = _unknowns_batch()

    reference = preprocessing.OrdinalEncoder(**params)
    reference.learn_many(FRAME_BACKENDS["pandas"]().frame(learn))
    expected = _read_rows(reference.transform_many(FRAME_BACKENDS["pandas"]().frame(trans)))

    encoder = preprocessing.OrdinalEncoder(**params)
    encoder.learn_many(frame_backend.frame(learn))
    got = _read_rows(encoder.transform_many(frame_backend.frame(trans)))

    assert got == expected


@pytest.mark.parametrize("params", list(ENCODER_PARAMS))
def test_transform_many_matches_transform_one(
    frame_backend: FrameBackend, params: dict[str, Any]
) -> None:
    """Each row of `transform_many` agrees with `transform_one`, including unknowns and nulls."""
    learn, trans, rows = _unknowns_batch()

    encoder = preprocessing.OrdinalEncoder(**params)
    encoder.learn_many(frame_backend.frame(learn))

    many = _read_rows(encoder.transform_many(frame_backend.frame(trans)))
    for row, many_row in zip(rows, many):
        assert encoder.transform_one(row) == many_row


@pytest.mark.parametrize("sentinel_kind", ["none", "nan", "pd_na"])
def test_transform_many_pandas_object_missing_maps_to_none_value(sentinel_kind: str) -> None:
    """In a pandas object column, ``None`` / ``NaN`` / ``pd.NA`` all encode to ``none_value``.

    The cross-backend tests already feed ``None`` through every backend (the nullable/pyarrow
    constructors turn it into ``pd.NA``); this pins the remaining pandas-native sentinels.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    sentinel: typing.Any = {"none": None, "nan": float("nan"), "pd_na": pd.NA}[sentinel_kind]
    X = pd.DataFrame({"c": ["a", sentinel, "b"]})

    encoder = preprocessing.OrdinalEncoder()
    encoder.learn_many(X)

    # The missing sentinel is never registered as a category, and encodes to none_value (-1).
    assert dict(encoder.values["c"]) == {"a": 1, "b": 2}
    assert _read_rows(encoder.transform_many(X)) == [{"c": 1}, {"c": -1}, {"c": 2}]


def test_transform_many_numeric_nan_maps_to_none_value(frame_backend: FrameBackend) -> None:
    """A numeric ``NaN`` is treated as missing on every backend.

    polars/pyarrow keep ``NaN`` distinct from null, so the encoder folds it into the missing
    mask explicitly; ``learn_many`` likewise skips it instead of learning it as a category.
    """
    data = {"c": [1.0, 2.0, float("nan"), 1.0]}

    encoder = preprocessing.OrdinalEncoder()
    encoder.learn_many(frame_backend.frame(data))

    assert dict(encoder.values["c"]) == {1.0: 1, 2.0: 2}
    expected = [{"c": 1}, {"c": 2}, {"c": -1}, {"c": 1}]
    assert _read_rows(encoder.transform_many(frame_backend.frame(data))) == expected


def test_transform_many_returns_native_backend(frame_backend: FrameBackend) -> None:
    """`transform_many` returns the input backend's native frame type."""
    data, _ = _categorical_batch()

    encoder = preprocessing.OrdinalEncoder()
    X = frame_backend.frame(data)
    encoder.learn_many(X)
    out = encoder.transform_many(X)

    assert type(out) is type(X)


def test_transform_many_preserves_pandas_index() -> None:
    """The pandas path keeps the input index on the encoded frame."""
    pytest.importorskip("pandas")
    import pandas as pd

    index = [7, 8, 9]
    X = pd.DataFrame({"c1": ["a", "b", "a"], "c2": ["x", "y", "z"]}, index=index)

    encoder = preprocessing.OrdinalEncoder()
    encoder.learn_many(X)
    out = encoder.transform_many(X)

    assert list(out.index) == index
