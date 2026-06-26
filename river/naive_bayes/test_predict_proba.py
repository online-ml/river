from __future__ import annotations

import typing

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from river import datasets, naive_bayes
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from river.conftest import FrameBackend


def _columnar(dataset, n: int = 80, cast=None):
    rows = list(dataset.take(n))
    feats = [x for x, _ in rows]
    targets = [y if cast is None else cast(y) for _, y in rows]
    cols = list(feats[0])
    data = {c: [f[c] for f in feats] for c in cols}
    return data, targets, feats


def _trained_gnb(data, targets):
    model = naive_bayes.GaussianNB()
    model.learn_many(pd.DataFrame(data), pd.Series(targets))
    return model


@pytest.fixture()
def dataset():
    return datasets.Bananas()


def test_predict_proba_many(frame_backend: FrameBackend, dataset):
    """`predict_proba_many` must yield identical probabilities on every backend."""
    data, targets, feats = _columnar(dataset, cast=bool)
    model = _trained_gnb(data, targets)
    pandas = FRAME_BACKENDS["pandas"]()
    ref = nw.from_native(model.predict_proba_many(pandas.frame(data)), eager_only=True).to_numpy()
    got = nw.from_native(
        model.predict_proba_many(frame_backend.frame(data)), eager_only=True
    ).to_numpy()
    assert np.allclose(got, ref, atol=1e-12)


def test_predict_proba_many_matches_sklearn(dataset):
    data, targets, _ = _columnar(dataset, cast=bool)
    X = pd.DataFrame(data)
    y = pd.Series(targets)
    river_model = _trained_gnb(data, targets)
    river_probs = nw.from_native(
        river_model.predict_proba_many(X),
        eager_only=True,
    ).to_pandas()
    skl_model = GaussianNB()
    skl_model.fit(X, y)
    skl_probs = pd.DataFrame(
        skl_model.predict_proba(X),
        columns=skl_model.classes_,
    )
    river_probs.columns = list(skl_probs.columns)
    assert np.allclose(
        river_probs.to_numpy(),
        skl_probs.to_numpy(),
        atol=1e-2,
    )
