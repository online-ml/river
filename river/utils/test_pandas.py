from __future__ import annotations

import pytest

from river import feature_extraction, linear_model, preprocessing, stats
from river.utils import pandas as pandas_utils


def _raise_missing_pandas() -> None:
    raise ImportError("`pandas` is required for this operation.")


def test_transform_many_requires_pandas_only_for_pandas_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `StandardScaler.transform_many` keeps a pandas fast path, so a pandas input still needs
    # pandas. `learn_many` is routed through narwhals and never imports pandas.
    import pandas as pd

    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)

    scaler = preprocessing.StandardScaler()
    scaler.learn_many(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    with pytest.raises(ImportError, match="pandas"):
        scaler.transform_many(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))


def test_transform_many_does_not_require_pandas_for_polars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A non-pandas backend takes the agnostic path, so `transform_many` must work even when
    # pandas is unavailable, returning the caller's own backend.
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)
    pl = pytest.importorskip("polars")

    scaler = preprocessing.StandardScaler()
    scaler.learn_many(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    out = scaler.transform_many(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))

    assert isinstance(out, pl.DataFrame)
    assert len(out) == 3


def test_predict_many_does_not_require_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    # `linear_model` mini-batching is routed through narwhals, so `predict_many` must work
    # on a non-pandas backend even when pandas is unavailable (see issues #1881 / #1805).
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)
    pl = pytest.importorskip("polars")

    model = linear_model.LinearRegression()
    model.learn_many(pl.DataFrame({"a": [1.0, 2.0, 3.0]}), pl.Series("y", [1.0, 2.0, 3.0]))
    out = model.predict_many(pl.DataFrame({"a": [1.0, 2.0, 3.0]}))

    assert isinstance(out, pl.Series)
    assert len(out) == 3


def test_optional_pandas_property_requires_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)

    agg = feature_extraction.Agg(on="value", by="group", how=stats.Mean())
    agg.learn_one({"group": "x", "value": 1})  # type: ignore[no-untyped-call]

    with pytest.raises(ImportError, match="pandas"):
        _ = agg.state
