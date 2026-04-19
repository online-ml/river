from __future__ import annotations

import pytest

from river import feature_extraction, linear_model, preprocessing, stats
from river.utils import pandas as pandas_utils


def _raise_missing_pandas():
    raise ImportError("`pandas` is required for this operation.")


def test_transform_many_requires_pandas(monkeypatch):
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)

    with pytest.raises(ImportError, match="pandas"):
        preprocessing.StandardScaler().transform_many(object())


def test_predict_many_requires_pandas(monkeypatch):
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)

    with pytest.raises(ImportError, match="pandas"):
        linear_model.LinearRegression().predict_many(object())


def test_optional_pandas_property_requires_pandas(monkeypatch):
    monkeypatch.setattr(pandas_utils, "import_pandas", _raise_missing_pandas)

    agg = feature_extraction.Agg(on="value", by="group", how=stats.Mean())
    agg.learn_one({"group": "x", "value": 1})

    with pytest.raises(ImportError, match="pandas"):
        _ = agg.state
