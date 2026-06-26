from __future__ import annotations

import math
import typing

import pytest
from sklearn import linear_model as sklm

from river import anomaly, datasets, optim
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from typing import Any

    from river.conftest import FrameBackend

tests = {
    "Vanilla": (
        {"optimizer": optim.SGD(1e-2), "nu": 0.5},
        {"learning_rate": "constant", "eta0": 1e-2, "nu": 0.5},
    ),
    "No intercept": (
        {"optimizer": optim.SGD(1e-2), "nu": 0.5, "intercept_lr": 0.0},
        {"learning_rate": "constant", "eta0": 1e-2, "nu": 0.5, "fit_intercept": False},
    ),
}


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    tests.values(),
    ids=tests.keys(),
)
def test_sklearn_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results."""

    rv = anomaly.OneClassSVM(**river_params)
    sk = sklm.SGDOneClassSVM(**sklearn_params)

    for x, _ in datasets.Phishing().take(100):
        rv.learn_one(x)
        sk.partial_fit([list(x.values())])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])


# Dataframe-agnostic mini-batching (narwhals): pandas, polars and pyarrow


def _columnar(n: int = 80) -> tuple[dict[str, list[Any]], list[dict[str, Any]]]:
    """Materialise a dataset into ({column: values}, [feature dicts])."""
    feats = [x for x, _ in datasets.Phishing().take(n)]
    cols = list(feats[0])
    data = {c: [f[c] for f in feats] for c in cols}
    return data, feats


def test_learn_many_is_backend_agnostic(frame_backend: FrameBackend) -> None:
    """`learn_many` must produce identical weights regardless of the input backend."""

    data, _ = _columnar()

    pandas = FRAME_BACKENDS["pandas"]()
    reference = anomaly.OneClassSVM()
    reference.learn_many(pandas.frame(data))

    model = anomaly.OneClassSVM()
    model.learn_many(frame_backend.frame(data))

    assert model.weights.keys() == reference.weights.keys()
    for key, weight in reference.weights.items():
        assert math.isclose(model.weights[key], weight, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(model.intercept, reference.intercept, rel_tol=1e-12, abs_tol=1e-12)


def test_learn_many_matches_learn_one(frame_backend: FrameBackend) -> None:
    """Row-by-row `learn_many` must match `learn_one` on every backend.

    With a single-row batch the mean gradient reduces to the one-sample gradient, so the two
    APIs are numerically identical -- but only because the `Hinge` gradient now resolves the
    margin boundary the same way in its scalar and vectorised branches (the very first sample
    sits exactly on the margin, since `intercept_init=1.0` and the weights start at zero).
    """

    data, feats = _columnar()

    one = anomaly.OneClassSVM()
    for x in feats:
        one.learn_one(x)

    many = anomaly.OneClassSVM()
    for i in range(len(feats)):
        many.learn_many(frame_backend.frame({c: [data[c][i]] for c in data}))

    for key in one.weights:
        assert math.isclose(many.weights[key], one.weights[key], rel_tol=1e-9)
    assert math.isclose(many.intercept, one.intercept, rel_tol=1e-9)


def test_learn_many_preserves_pandas_index() -> None:
    """A pandas input still learns correctly when it carries a non-default index."""
    import pandas as pd

    data, _ = _columnar(20)
    index = list(range(100, 100 + len(next(iter(data.values())))))

    default = anomaly.OneClassSVM()
    default.learn_many(pd.DataFrame(data))

    reindexed = anomaly.OneClassSVM()
    reindexed.learn_many(pd.DataFrame(data, index=index))

    for key, weight in default.weights.items():
        assert math.isclose(reindexed.weights[key], weight, rel_tol=1e-12, abs_tol=1e-12)
