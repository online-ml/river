from __future__ import annotations

import typing

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd

from river import datasets, linear_model, metrics, multiclass, preprocessing, stream

if typing.TYPE_CHECKING:
    from river.conftest import FrameBackend


def test_online_batch_consistent():
    # Batch

    batch = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression()
    )

    dataset = datasets.ImageSegments()

    batch_metric = metrics.MacroF1()

    for i, x in enumerate(pd.read_csv(dataset.path, chunksize=1)):
        y = x.pop("category")
        y_pred = batch.predict_many(x)
        batch.learn_many(x, y)

        for yt, yp in zip(y, y_pred):
            if yp is not None:
                batch_metric.update(yt, yp)

        if i == 30:
            break

    # Online

    online = preprocessing.StandardScaler() | multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression()
    )

    online_metric = metrics.MacroF1()

    X = pd.read_csv(dataset.path)
    Y = X.pop("category")

    for i, (x, y) in enumerate(stream.iter_frame(X, Y)):
        y_pred = online.predict_one(x)
        online.learn_one(x, y)

        if y_pred is not None:
            online_metric.update(y, y_pred)

        if i == 30:
            break

    assert online_metric.get() == batch_metric.get()


# `OneVsRestClassifier`'s mini-batch methods are routed through narwhals. These tests pin that
# `learn_many`/`predict_many`/`predict_proba_many` behave identically on every dataframe backend,
# using the pandas path as the oracle.


def _multiclass_batch(n: int = 90) -> tuple[dict[str, list[float]], list[str]]:
    """A reproducible, linearly-separable-ish 3-class batch as (feature columns, labels)."""
    rng = np.random.RandomState(42)
    classes = ["cat", "dog", "fish"]
    y = [classes[i % 3] for i in range(n)]
    offset = {"cat": 0.0, "dog": 4.0, "fish": -4.0}
    data = {
        "a": [rng.normal(offset[label], 1.0) for label in y],
        "b": [rng.normal(-offset[label], 1.0) for label in y],
    }
    return data, y


def test_ovr_predict_many_backend_agnostic(frame_backend: FrameBackend) -> None:
    """`predict_many` returns the same labels (and series name) on every backend."""
    data, y = _multiclass_batch()

    reference = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    reference.learn_many(pd.DataFrame(data), pd.Series(y, name="category"))
    expected = nw.from_native(
        reference.predict_many(pd.DataFrame(data)), series_only=True
    ).to_list()

    model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    model.learn_many(frame_backend.frame(data), frame_backend.series(y, name="category"))
    got = nw.from_native(model.predict_many(frame_backend.frame(data)), series_only=True).to_list()

    assert got == expected


def test_ovr_predict_proba_many_backend_agnostic(frame_backend: FrameBackend) -> None:
    """`predict_proba_many` yields the same per-class probabilities on every backend."""
    data, y = _multiclass_batch()

    reference = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    reference.learn_many(pd.DataFrame(data), pd.Series(y, name="category"))
    expected = nw.from_native(reference.predict_proba_many(pd.DataFrame(data)), eager_only=True)

    model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    model.learn_many(frame_backend.frame(data), frame_backend.series(y, name="category"))
    got = nw.from_native(model.predict_proba_many(frame_backend.frame(data)), eager_only=True)

    # Columns are the class labels; non-pandas backends stringify them, so align by string name.
    for label in expected.columns:
        got_col = np.asarray(got[str(label)].to_numpy(), dtype=float)
        np.testing.assert_allclose(
            got_col, np.asarray(expected[label].to_numpy(), dtype=float), atol=1e-9
        )


def test_ovr_predict_many_returns_native_backend(frame_backend: FrameBackend) -> None:
    """The prediction series is rebuilt in the caller's own backend."""
    data, y = _multiclass_batch()
    native = frame_backend.frame(data)

    model = multiclass.OneVsRestClassifier(linear_model.LogisticRegression())
    model.learn_many(native, frame_backend.series(y, name="category"))
    out = model.predict_many(native)

    assert type(out) is type(frame_backend.series(y, name="category"))
