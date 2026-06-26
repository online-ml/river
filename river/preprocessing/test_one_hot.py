from __future__ import annotations

import random
import string
import typing

import narwhals.stable.v2 as nw
import pytest

from river import preprocessing
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame

    from river.conftest import FrameBackend

# `transform_many` mini-batching is routed through narwhals: pandas keeps the historical sparse
# fast path, while every other backend is encoded densely via `narwhals.Series.to_dummies`. These
# tests pin the cross-backend behaviour (values, native return type, pandas index), using the
# pandas path as the oracle.

CONFIGS: list[dict[str, typing.Any]] = [
    {},
    {"drop_zeros": True},
    {"drop_first": True},
    {"drop_zeros": True, "drop_first": True},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}, "drop_zeros": True},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}, "drop_first": True},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}, "drop_zeros": True, "drop_first": True},
]

# Backends whose dtypes round-trip the input unchanged, so their encoded output must equal the
# pandas `get_dummies` reference exactly (including missing values). The `pandas[nullable]` /
# `pandas[pyarrow]` variants are excluded on purpose: `convert_dtypes` re-represents the data
# (`1.0` -> `1`, `None` -> `pd.NA`), which legitimately changes value-derived column names.
NATIVE_NON_PANDAS_BACKENDS = ["polars", "pyarrow"]


def _categorical_batch(n: int = 40) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Build a reproducible batch of two categorical columns as (columns, row dicts)."""
    rng = random.Random(42)
    alphabet = list(string.ascii_lowercase[:6])
    rows = [{"c1": rng.choice(alphabet), "c2": rng.choice(alphabet)} for _ in range(n)]
    data = {"c1": [r["c1"] for r in rows], "c2": [r["c2"] for r in rows]}
    return data, rows


def _rows(native: IntoDataFrame) -> list[dict[str, int]]:
    """Read a native one-hot frame back into a list of ``{column: int}`` rows via narwhals."""
    frame = nw.from_native(native, eager_only=True)
    return [{col: int(row[col]) for col in frame.columns} for row in frame.iter_rows(named=True)]


def _assert_pandas_parity(
    backend: FrameBackend,
    config: dict[str, typing.Any],
    learn: dict[str, list[typing.Any]],
    transform: dict[str, list[typing.Any]] | None = None,
) -> list[dict[str, int]]:
    """Encode on `backend` and assert it matches the pandas reference; return the encoded rows.

    Both encoders run the same `config`: they learn on `learn`, then encode `transform`
    (defaulting to `learn`). The pandas path is the oracle, so any divergence is a backend bug.
    """
    transform = learn if transform is None else transform

    reference = preprocessing.OneHotEncoder(**config)
    reference.learn_many(FRAME_BACKENDS["pandas"]().frame(learn))
    expected = _rows(reference.transform_many(FRAME_BACKENDS["pandas"]().frame(transform)))

    encoder = preprocessing.OneHotEncoder(**config)
    encoder.learn_many(backend.frame(learn))
    got = _rows(encoder.transform_many(backend.frame(transform)))

    assert got == expected
    return got


@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_transform_many_is_backend_agnostic(
    frame_backend: FrameBackend, config: dict[str, typing.Any]
) -> None:
    """`transform_many` yields identical values on every backend (column set included).

    Reading rows back via `_rows` keys on each frame's column set, so this also pins that explicit
    `categories` restrict the output columns identically across backends, like scikit-learn.
    """
    data, _ = _categorical_batch()
    _assert_pandas_parity(frame_backend, config, data)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"drop_first": True},
        {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}},
        {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}, "drop_first": True},
    ],
    ids=str,
)
def test_transform_many_matches_transform_one(
    frame_backend: FrameBackend, config: dict[str, typing.Any]
) -> None:
    """Each row of `transform_many` agrees with `transform_one` on every backend.

    Pinned with ``drop_zeros=False`` so both paths materialise the full column set, which makes
    ``transform_one``'s per-row ``min`` (used by ``drop_first``) equal the single global column
    ``transform_many`` drops. With ``drop_zeros=True`` the present-only keys vary per row, so the
    two paths legitimately diverge for ``drop_first``; that combination is out of scope here.
    """
    data, rows = _categorical_batch()

    encoder = preprocessing.OneHotEncoder(drop_zeros=False, **config)
    encoder.learn_many(frame_backend.frame(data))

    many = _rows(encoder.transform_many(frame_backend.frame(data)))
    for row, many_row in zip(rows, many):
        one = encoder.transform_one(row)
        for key in set(one) | set(many_row):
            assert one.get(key, 0) == many_row.get(key, 0), key


@pytest.mark.parametrize("backend_name", NATIVE_NON_PANDAS_BACKENDS)
@pytest.mark.parametrize(
    "config",
    [{}, {"drop_zeros": True}, {"drop_first": True}, {"drop_zeros": True, "drop_first": True}],
    ids=str,
)
def test_transform_many_missing_matches_pandas(
    backend_name: str, config: dict[str, typing.Any]
) -> None:
    """Missing values (string ``None``, numeric ``NaN``) encode identically to the pandas path.

    ``get_dummies`` omits both from the encoded ``1``s; polars/pyarrow keep ``NaN`` distinct from
    null, so the encoder folds ``NaN`` into null to match pandas exactly.
    """
    pytest.importorskip(backend_name)
    data: dict[str, list[typing.Any]] = {
        "c1": ["a", None, "b", "a"],
        "c2": [1.0, 2.0, float("nan"), 1.0],
    }
    _assert_pandas_parity(FRAME_BACKENDS[backend_name](), config, data)


def test_transform_many_missing_is_all_zeros_pandas() -> None:
    """Pin the pandas contract: a missing cell encodes to all-zeros (no ``1`` in any dummy).

    ``None`` is still registered as the literal ``"None"`` category, so a zero-filled ``c_None``
    column is padded in (mirroring ``get_dummies``, which omits the missing row's ``1``).
    """
    import pandas as pd

    X = pd.DataFrame({"c": ["a", None, "b"]})
    encoder = preprocessing.OneHotEncoder(drop_zeros=False)
    encoder.learn_many(X)

    assert _rows(encoder.transform_many(X)) == [
        {"c_None": 0, "c_a": 1, "c_b": 0},
        {"c_None": 0, "c_a": 0, "c_b": 0},
        {"c_None": 0, "c_a": 0, "c_b": 1},
    ]


def test_transform_many_pads_categories_from_earlier_batches(frame_backend: FrameBackend) -> None:
    """Categories seen in an earlier `learn_many` but absent from the transform batch are padded.

    With ``drop_zeros=False`` the encoder re-emits every previously-seen category as an all-zero
    column. The agnostic tests learn and transform the same batch (nothing needs padding); this one
    grows the vocabulary first to exercise zero-column padding against a larger prior vocabulary.
    """
    learn = {"c1": ["a", "b", "c"], "c2": ["x", "y", "z"]}
    trans = {"c1": ["a", "a", "b"], "c2": ["x", "y", "x"]}  # "c" and "z" never appear here

    got = _assert_pandas_parity(frame_backend, {"drop_zeros": False}, learn, trans)
    # The categories unseen in this batch are re-emitted as all-zero columns on every backend.
    assert all(row["c1_c"] == 0 and row["c2_z"] == 0 for row in got)


def test_transform_many_returns_native_backend(frame_backend: FrameBackend) -> None:
    """`transform_many` returns the input backend's native frame type."""
    data, _ = _categorical_batch()
    X = frame_backend.frame(data)
    encoder = preprocessing.OneHotEncoder(drop_zeros=True)
    encoder.learn_many(X)
    out = encoder.transform_many(X)

    # The pandas fast path always returns a classic pandas frame, regardless of the input's pandas
    # dtype backend (numpy / nullable / pyarrow), so compare the top-level module only.
    assert type(out) is type(X)


def test_transform_many_pandas_is_sparse_others_dense() -> None:
    """Pandas keeps the sparse fast path; non-pandas backends emit dense integer columns."""
    import pandas as pd

    data, _ = _categorical_batch()

    pandas_out = preprocessing.OneHotEncoder(drop_zeros=True).transform_many(pd.DataFrame(data))
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in pandas_out.dtypes)

    pl = pytest.importorskip("polars")
    polars_out = preprocessing.OneHotEncoder(drop_zeros=True).transform_many(pl.DataFrame(data))
    assert all(dtype.is_integer() for dtype in polars_out.schema.dtypes())


def test_transform_many_preserves_pandas_index() -> None:
    """The pandas fast path keeps the input index on the encoded frame."""
    import pandas as pd

    index = [100, 200, 300]
    X = pd.DataFrame({"c1": ["a", "b", "a"], "c2": ["x", "y", "z"]}, index=index)

    encoder = preprocessing.OneHotEncoder(drop_zeros=False)
    encoder.learn_many(X)

    assert list(encoder.transform_many(X).index) == index


@pytest.mark.parametrize(
    "config",
    [
        {"categories": {"c1": {"z"}}, "drop_zeros": True},  # no configured category is present
        {"drop_zeros": True, "drop_first": True},  # the lone dummy is dropped as the "first"
    ],
    ids=str,
)
def test_transform_many_empty_result_has_no_columns(
    frame_backend: FrameBackend, config: dict[str, typing.Any]
) -> None:
    """A config that encodes nothing returns a column-less frame on every backend, without raising.

    polars/pyarrow cannot represent an N-row, 0-column frame, so they collapse to 0 rows while
    pandas keeps the input rows; "no columns" is the invariant every backend can honour, and
    reaching the assertion pins that the trailing empty ``select`` does not raise.
    """
    data = {"c1": ["a", "a", "a"]}
    encoder = preprocessing.OneHotEncoder(**config)
    encoder.learn_many(frame_backend.frame(data))
    out = nw.from_native(encoder.transform_many(frame_backend.frame(data)), eager_only=True)

    assert list(out.columns) == []
