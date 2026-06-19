from __future__ import annotations

import copy
import itertools
import math
import random
import typing

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from sklearn import linear_model as sklm
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss

from river import datasets, optim, preprocessing, stream, utils
from river import linear_model as lm
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from river.base.typing import FeatureName
    from river.conftest import FrameBackend
    from river.datasets.base import Dataset

    # ({column: values}, [targets], [feature dicts]) as produced by `_columnar`.
    Columnar = tuple[dict[FeatureName, list[Any]], list[Any], list[dict[FeatureName, Any]]]


def _pd_split(df, n):
    """Split a pandas DataFrame or Series into n chunks without triggering swapaxes deprecation."""
    indices = np.array_split(range(len(df)), n)
    return [df.iloc[idx] for idx in indices]


def iter_perturbations(keys, n=10):
    """Enumerate perturbations that will be applied to the weights."""

    # Enumerate unit vectors
    for i in keys:
        yield {j: int(i == j) for j in keys}

    # Enumerate samples from a spherical Gaussian
    for _ in range(n):
        p = {j: random.gauss(0, 1) for j in keys}
        norm = utils.math.norm(p, order=2)
        for j in p:
            p[j] /= norm
        yield p


@pytest.mark.parametrize(
    "lm, dataset",
    [
        pytest.param(
            lm(optimizer=copy.deepcopy(optimizer), initializer=initializer, l2=0),
            dataset,
            id=f"{lm.__name__} - {optimizer} - {initializer}",
        )
        for lm, dataset in [
            (lm.LinearRegression, datasets.TrumpApproval().take(100)),
            (lm.LogisticRegression, datasets.Bananas().take(100)),
        ]
        for optimizer, initializer in itertools.product(
            [
                optim.AdaBound(),
                optim.AdaDelta(),
                optim.AdaGrad(),
                optim.AdaMax(),
                optim.Adam(),
                optim.AMSGrad(),
                # TODO: check momentum optimizers
                # optim.Momentum(),
                # optim.NesterovMomentum(),
                optim.RMSProp(),
                optim.SGD(),
            ],
            [
                optim.initializers.Zeros(),
                optim.initializers.Normal(mu=0, sigma=1, seed=42),
            ],
        )
    ],
)
def test_finite_differences(lm, dataset):
    """Checks the gradient of a linear model via finite differences.

    References
    ----------
    [^1]: [How to test gradient implementations](https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/)
    [^2]: [Stochastic Gradient Descent Tricks](https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf)

    """

    scaler = preprocessing.StandardScaler()
    eps = 1e-6

    for x, y in dataset:
        scaler.learn_one(x)
        x = scaler.transform_one(x)

        # Store the current gradient and weights
        gradient, _ = lm._eval_gradient_one(x, y, 1)
        weights = copy.deepcopy(lm._weights)

        # d is a set of weight perturbations
        for d in iter_perturbations(weights.keys()):
            # Perturb the weights and obtain the loss with the new weights
            lm._weights = utils.VectorDict({i: weights[i] + eps * di for i, di in d.items()})
            forward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))
            lm._weights = utils.VectorDict({i: weights[i] - eps * di for i, di in d.items()})
            backward = lm.loss(y_true=y, y_pred=lm._raw_dot_one(x))

            # We expect g and h to be equal
            g = utils.math.dot(d, gradient)
            h = (forward - backward) / (2 * eps)

            # Compare signs
            # TODO: reactivate this check
            # assert np.sign(g) == np.sign(h)

            # Check absolute difference
            # TODO: decrease the tolerance
            assert abs(g - h) < 1e-5

        # Reset the weights to their original values in order not to influence
        # the training loop, even though it doesn't really matter.
        lm._weights = weights
        lm.learn_one(x, y)


def test_one_many_consistent():
    """Checks that using learn_one or learn_many produces the same result."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    one = lm.LinearRegression()
    for x, y in stream.iter_frame(X[:100], Y[:100]):
        one.learn_one(x, y)

    many = lm.LinearRegression()
    for xb, yb in zip(_pd_split(X[:100], len(X[:100])), _pd_split(Y[:100], len(Y[:100]))):
        many.learn_many(xb, yb)

    for i in X:
        assert math.isclose(one.weights[i], many.weights[i])


def test_shuffle_columns():
    """Checks that learn_many works identically whether columns are shuffled or not."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    normal = lm.LinearRegression()
    for xb, yb in zip(_pd_split(X, 10), _pd_split(Y, 10)):
        normal.learn_many(xb, yb)

    shuffled = lm.LinearRegression()
    for xb, yb in zip(_pd_split(X, 10), _pd_split(Y, 10)):
        cols = np.random.permutation(X.columns)
        shuffled.learn_many(xb[cols], yb)

    for i in X:
        assert math.isclose(normal.weights[i], shuffled.weights[i])


def test_add_remove_columns():
    """Checks that no exceptions are raised whenever columns are dropped and/or added."""

    X = pd.read_csv(datasets.TrumpApproval().path)
    Y = X.pop("five_thirty_eight")

    lin_reg = lm.LinearRegression()
    for xb, yb in zip(_pd_split(X, 10), _pd_split(Y, 10)):
        # Pick half of the columns at random
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        lin_reg.learn_many(xb[cols], yb)


class ScikitLearnSquaredLoss:
    """sklearn removes the leading 2 from the gradient of the squared loss."""

    def gradient(self, y_true, y_pred):
        return y_pred - y_true


lin_reg_tests = {
    "Vanilla": (
        {"optimizer": optim.SGD(1e-2), "loss": ScikitLearnSquaredLoss()},
        {"learning_rate": "constant", "eta0": 1e-2, "alpha": 0},
    ),
    "Huber": (
        {"optimizer": optim.SGD(1e-2), "loss": optim.losses.Huber()},
        {"loss": "huber", "learning_rate": "constant", "eta0": 1e-2, "alpha": 0},
    ),
    "No intercept": (
        {
            "optimizer": optim.SGD(1e-2),
            "intercept_lr": 0,
            "loss": ScikitLearnSquaredLoss(),
        },
        {
            "learning_rate": "constant",
            "eta0": 1e-2,
            "alpha": 0,
            "fit_intercept": False,
        },
    ),
    "L2 regu": (
        {
            "optimizer": optim.SGD(1e-2),
            "loss": ScikitLearnSquaredLoss(),
            "l2": 1e-3,
        },
        {"learning_rate": "constant", "eta0": 1e-2, "alpha": 1e-3},
    ),
}


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    lin_reg_tests.values(),
    ids=lin_reg_tests.keys(),
)
def test_lin_reg_sklearn_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results."""

    ss = preprocessing.StandardScaler()
    rv = lm.LinearRegression(**river_params)
    sk = sklm.SGDRegressor(**sklearn_params)

    for x, y in datasets.TrumpApproval().take(100):
        ss.learn_one(x)
        x = ss.transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])

    assert math.isclose(rv.intercept, sk.intercept_[0])


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    lin_reg_tests.values(),
    ids=lin_reg_tests.keys(),
)
def test_lin_reg_sklearn_learn_many_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results
    when learn_many is used."""

    ss = preprocessing.StandardScaler()
    rv = lm.LinearRegression(**river_params)
    sk = sklm.SGDRegressor(**sklearn_params)

    for x, y in datasets.TrumpApproval().take(100):
        ss.learn_one(x)
        x = ss.transform_one(x)
        rv.learn_many(pd.DataFrame([x]), pd.Series([y]))
        sk.partial_fit([list(x.values())], [y])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[i])

    assert math.isclose(rv.intercept, sk.intercept_[0])


log_reg_tests = {
    "Vanilla": (
        {"optimizer": optim.SGD(1e-2)},
        {"learning_rate": "constant", "eta0": 1e-2, "alpha": 0, "loss": "log_loss"},
    ),
    "Hinge": (
        {"optimizer": optim.SGD(1e-2), "loss": optim.losses.Hinge()},
        {"learning_rate": "constant", "eta0": 1e-2, "alpha": 0},
    ),
    "No intercept": (
        {"optimizer": optim.SGD(1e-2), "intercept_lr": 0},
        {
            "learning_rate": "constant",
            "eta0": 1e-2,
            "alpha": 0,
            "loss": "log_loss",
            "fit_intercept": False,
        },
    ),
    "L2 regu": (
        {
            "optimizer": optim.SGD(1e-2),
            "l2": 1e-3,
        },
        {
            "learning_rate": "constant",
            "eta0": 1e-2,
            "alpha": 1e-3,
            "loss": "log_loss",
        },
    ),
    "Inverse-scaling": (
        {
            "optimizer": optim.SGD(optim.schedulers.InverseScaling(1e-2)),
            "intercept_lr": optim.schedulers.InverseScaling(1e-2),
        },
        {"eta0": 1e-2, "alpha": 0, "learning_rate": "invscaling", "loss": "log_loss"},
    ),
    "Optimal": (
        {
            "optimizer": optim.SGD(optim.schedulers.Optimal(optim.losses.Hinge(), alpha=1e-3)),
            "loss": optim.losses.Hinge(),
            "intercept_lr": optim.schedulers.Optimal(optim.losses.Hinge(), alpha=1e-3),
            "l2": 1e-3,
        },
        {"learning_rate": "optimal", "alpha": 1e-3},
    ),
    "Optimal no intercept": (
        {
            "optimizer": optim.SGD(optim.schedulers.Optimal(optim.losses.Hinge(), alpha=1e-3)),
            "loss": optim.losses.Hinge(),
            "intercept_lr": 0,
            "l2": 1e-3,
        },
        {"learning_rate": "optimal", "alpha": 1e-3, "fit_intercept": False},
    ),
}


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    log_reg_tests.values(),
    ids=log_reg_tests.keys(),
)
def test_log_reg_sklearn_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results."""

    ss = preprocessing.StandardScaler()
    rv = lm.LogisticRegression(**river_params)
    sk = sklm.SGDClassifier(**sklearn_params)

    for x, y in datasets.Bananas().take(100):
        ss.learn_one(x)
        x = ss.transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])

    assert math.isclose(rv.intercept, sk.intercept_[0])


perceptron_tests = {
    "Vanilla": (
        {},
        {},
    ),
    "L2 regu": (
        {
            "l2": 1e-3,
        },
        {
            "alpha": 1e-3,
            "penalty": "l2",
        },
    ),
}


@pytest.mark.parametrize(
    "river_params, sklearn_params",
    perceptron_tests.values(),
    ids=perceptron_tests.keys(),
)
def test_perceptron_sklearn_coherence(river_params, sklearn_params):
    """Checks that the sklearn and river implementations produce the same results."""

    ss = preprocessing.StandardScaler()
    rv = lm.Perceptron(**river_params)
    sk = sklm.Perceptron(**sklearn_params)

    for x, y in datasets.Bananas().take(100):
        ss.learn_one(x)
        x = ss.transform_one(x)
        rv.learn_one(x, y)
        sk.partial_fit([list(x.values())], [y], classes=[False, True])

    for i, w in enumerate(rv.weights.values()):
        assert math.isclose(w, sk.coef_[0][i])

    assert math.isclose(rv.intercept, sk.intercept_[0])


def test_lin_reg_sklearn_l1_non_regression():
    """Checks that the river L1 implementation results are no worse than sklearn L1."""

    X, y, true_coeffs = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=4,
        coef=True,
        random_state=273,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    ss = preprocessing.StandardScaler()

    rv = lm.LinearRegression(
        **{
            "optimizer": optim.SGD(1e-2),
            "loss": ScikitLearnSquaredLoss(),
            "l1": 1e-1,
        }
    )

    sk = sklm.SGDRegressor(
        **{
            "learning_rate": "constant",
            "eta0": 1e-2,
            "alpha": 1e-1,
            "penalty": "l1",
        }
    )

    for xi, yi in stream.iter_frame(X, y):
        ss.learn_one(xi)
        xi_tr = ss.transform_one(xi)
        rv.learn_one(xi_tr, yi)
        sk.partial_fit([list(xi_tr.values())], [yi])

    rv_coeffs = np.array(list(rv.weights.values()))
    sk_coeffs = sk.coef_

    # check that the shrinkage to zero is no worse than in sklearn
    assert np.sum(rv_coeffs > 0) <= np.sum(sk_coeffs > 0)

    # check that the river coefficients shrink to the true ones with 10% relative tolerance
    assert np.isclose(rv_coeffs, true_coeffs, rtol=5e-02, atol=0.0).all()


def test_log_reg_sklearn_l1_non_regression():
    """Checks that the river L1 implementation results are no worse than sklearn L1."""

    (
        X,
        y,
    ) = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=4,
        n_classes=2,
        random_state=273,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    ss = preprocessing.StandardScaler()

    rv = lm.LogisticRegression(
        **{
            "optimizer": optim.SGD(1e-2),
            "l1": 1e-3,
        }
    )

    sk = sklm.SGDClassifier(
        **{
            "learning_rate": "constant",
            "eta0": 1e-2,
            "alpha": 1e-3,
            "penalty": "l1",
            "loss": "log_loss",
        }
    )

    rv_pred = list()
    sk_pred = list()
    for xi, yi in stream.iter_frame(X, y):
        ss.learn_one(xi)
        xi_tr = ss.transform_one(xi)
        rv.learn_one(xi_tr, yi)
        sk.partial_fit([list(xi_tr.values())], [yi], classes=[False, True])

        rv_pred.append(rv.predict_one(xi_tr))
        sk_pred.append(sk.predict([list(xi_tr.values())])[0])

    rv_coeffs = np.array(list(rv.weights.values()))
    sk_coeffs = sk.coef_

    # check that the shrinkage to zero is no worse than in sklearn
    assert np.sum(rv_coeffs > 0) <= np.sum(sk_coeffs > 0)

    # since we can't access true coefficients from sklearn.make_classification, we compare losses
    assert math.isclose(log_loss(y, rv_pred), log_loss(y, sk_pred))


# Dataframe-agnostic mini-batching (narwhals): pandas, polars and pyarrow


def _columnar(dataset: Dataset, n: int = 80, cast: Callable[[Any], Any] | None = None) -> Columnar:
    """Materialise a dataset into ({column: values}, [targets], [feature dicts])."""
    rows = list(dataset.take(n))
    feats = [x for x, _ in rows]
    targets = [y if cast is None else cast(y) for _, y in rows]
    cols = list(feats[0])
    data = {c: [f[c] for f in feats] for c in cols}
    return data, targets, feats


@pytest.fixture(scope="module")
def reg_batch() -> Columnar:
    return _columnar(datasets.TrumpApproval(), cast=float)


@pytest.fixture(scope="module")
def clf_batch() -> Columnar:
    return _columnar(datasets.Bananas(), cast=bool)


@pytest.mark.parametrize("Model", [lm.LinearRegression, lm.LogisticRegression])
def test_learn_many_is_backend_agnostic(
    frame_backend: FrameBackend,
    Model: type[lm.LinearRegression | lm.LogisticRegression],
    reg_batch: Columnar,
    clf_batch: Columnar,
) -> None:
    """`learn_many` must produce identical weights regardless of the input backend."""

    data, targets, _ = reg_batch if Model is lm.LinearRegression else clf_batch

    pandas = FRAME_BACKENDS["pandas"]()
    reference = Model()
    reference.learn_many(pandas.frame(data), pandas.series(targets))

    model = Model()
    model.learn_many(frame_backend.frame(data), frame_backend.series(targets))

    assert model.weights.keys() == reference.weights.keys()
    for key, weight in reference.weights.items():
        assert math.isclose(model.weights[key], weight, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(model.intercept, reference.intercept, rel_tol=1e-12, abs_tol=1e-12)


def test_learn_many_matches_learn_one(frame_backend: FrameBackend, reg_batch: Columnar) -> None:
    """Row-by-row `learn_many` must match `learn_one` on every backend."""

    data, targets, feats = reg_batch

    one = lm.LinearRegression()
    for x, y in zip(feats, targets):
        one.learn_one(x, y)

    many = lm.LinearRegression()
    for i, y in enumerate(targets):
        X = frame_backend.frame({c: [data[c][i]] for c in data})
        many.learn_many(X, frame_backend.series([y]))

    for key in one.weights:
        assert math.isclose(many.weights[key], one.weights[key], rel_tol=1e-9)


def test_predict_many_returns_native_backend(
    frame_backend: FrameBackend, reg_batch: Columnar
) -> None:
    """`predict_many` returns the input backend's native type, with matching values."""

    data, targets, feats = reg_batch
    X = frame_backend.frame(data)

    model = lm.LinearRegression()
    model.learn_many(X, frame_backend.series(targets))

    y_pred = model.predict_many(X)
    assert type(y_pred).__module__.split(".")[0] == frame_backend.name

    expected = [model.predict_one(x) for x in feats]
    got = nw.from_native(y_pred, series_only=True).to_list()
    assert all(math.isclose(g, e, rel_tol=1e-9) for g, e in zip(got, expected))


def test_logreg_predict_many_matches_predict_one(
    frame_backend: FrameBackend, clf_batch: Columnar
) -> None:
    """Logistic `predict_many` labels must match `predict_one` on every backend."""

    data, targets, feats = clf_batch
    X = frame_backend.frame(data)

    model = lm.LogisticRegression()
    model.learn_many(X, frame_backend.series(targets))

    y_pred = model.predict_many(X)
    assert type(y_pred).__module__.split(".")[0] == frame_backend.name

    expected = [model.predict_one(x) for x in feats]
    got = nw.from_native(y_pred, series_only=True).to_list()
    assert [bool(g) for g in got] == [bool(e) for e in expected]


def test_logreg_predict_proba_many_matches_predict_proba_one(
    frame_backend: FrameBackend, clf_batch: Columnar
) -> None:
    """Logistic `predict_proba_many` must agree column-wise with `predict_proba_one`.

    Column order is always (False, True); only the labels differ across backends
    (booleans for pandas, their string form elsewhere).
    """

    data, targets, feats = clf_batch
    X = frame_backend.frame(data)

    model = lm.LogisticRegression()
    model.learn_many(X, frame_backend.series(targets))

    proba = model.predict_proba_many(X)
    assert type(proba).__module__.split(".")[0] == frame_backend.name

    expected_labels: list[bool | str] = (
        [False, True] if frame_backend.name == "pandas" else ["False", "True"]
    )
    proba_nw = nw.from_native(proba, eager_only=True)
    assert list(proba_nw.columns) == expected_labels

    got = proba_nw.to_numpy()
    for i, x in enumerate(feats):
        one = model.predict_proba_one(x)
        assert math.isclose(got[i, 0], one[False], rel_tol=1e-9)
        assert math.isclose(got[i, 1], one[True], rel_tol=1e-9)


def test_predict_many_preserves_pandas_index() -> None:
    """The pandas index contract is kept for series and frame outputs."""

    index = [100, 200, 300, 400]
    X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [4.0, 3, 2, 1]}, index=index)

    reg = lm.LinearRegression()
    reg.learn_many(X, pd.Series([1.0, 2, 3, 4], index=index, name="target"))
    pred_reg: pd.Series[Any] = reg.predict_many(X)
    assert list(pred_reg.index) == index
    assert pred_reg.name == "target"

    clf = lm.LogisticRegression()
    clf.learn_many(X, pd.Series([False, True, False, True], index=index, name="label"))
    proba: pd.DataFrame = clf.predict_proba_many(X)
    assert list(proba.index) == index
    assert list(proba.columns) == [False, True]

    pred_clf: pd.Series[Any] = clf.predict_many(X)
    assert list(pred_clf.index) == index
