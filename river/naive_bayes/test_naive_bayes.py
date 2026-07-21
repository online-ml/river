from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import naive_bayes as sk_naive_bayes

from river import compose, feature_extraction, naive_bayes


def river_models():
    """List of Naive Bayes models to test."""
    yield from [
        naive_bayes.MultinomialNB,
        naive_bayes.BernoulliNB,
        naive_bayes.ComplementNB,
    ]


def sklearn_models():
    """Mapping between Naives Bayes river models and sklearn models."""
    yield from [
        (naive_bayes.MultinomialNB, sk_naive_bayes.MultinomialNB),
        (naive_bayes.BernoulliNB, sk_naive_bayes.BernoulliNB),
        (naive_bayes.ComplementNB, sk_naive_bayes.ComplementNB),
    ]


def yield_dataset():
    """Incremental dataset."""
    yield from [
        ("Chinese Beijing Chinese", "yes"),
        ("Chinese Chinese Shanghai", "yes"),
        ("Chinese Macao", "yes"),
        ("Tokyo Japan Chinese", "no"),
    ]


def yield_batch_dataset():
    """Batch dataset."""
    for x, y in yield_dataset():
        yield pd.Series([x]), pd.Series([y])


def yield_unseen_data():
    yield from [
        "Chinese Beijing Chinese",
        "Chinese Chinese Shanghai",
        "Chinese Macao",
        "Tokyo Japan Chinese",
        "new unseen data",
        "Taiwanese Taipei",
        "Chinese ShanghaiShanghai",
        "Chinese",
        "Tokyo Macao",
        "Tokyo Tokyo",
        "Macao Macao new",
        "new",
    ]


def yield_batch_unseen_data():
    yield from [pd.Series(x) for x in yield_unseen_data()]


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            id=f"{model.__name__} - {alpha}",
        )
        for model in river_models()
        for alpha in [alpha for alpha in range(1, 4)]
    ],
)
def test_learn_one_methods(model):
    """Assert that the methods of the Naives Bayes class behave correctly."""
    assert model.predict_proba_one("not fitted yet") == {}
    assert model.predict_one("not fitted yet") is None

    for x, y in yield_dataset():
        model.learn_one(x, y)

    # Check class methods.
    if isinstance(model["model"], naive_bayes.ComplementNB) or isinstance(
        model["model"], naive_bayes.MultinomialNB
    ):
        assert model["model"]._more_tags() == {"positive input"}
    assert model["model"]._multiclass


@pytest.mark.parametrize(
    "model, batch_model",
    [
        pytest.param(
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            id=f"{model.__name__} - {alpha}",
        )
        for model in river_models()
        for alpha in [alpha for alpha in range(1, 4)]
    ],
)
def test_learn_many_vs_learn_one(model, batch_model):
    """Assert that the Naive Bayes river models provide the same results when learning in
    incremental and mini-batch modes. The models tested are MultinomialNB, BernoulliNB and
    ComplementNB with different alpha parameters..
    """
    for x, y in yield_dataset():
        model.learn_one(x, y)

    for x, y in yield_batch_dataset():
        batch_model.learn_many(x, y)

    assert model["model"].p_class("yes") == batch_model["model"].p_class("yes")
    assert model["model"].p_class("no") == batch_model["model"].p_class("no")

    # Assert batch and incremental models give same results
    for x, x_batch in zip(yield_unseen_data(), yield_batch_unseen_data()):
        assert model.predict_proba_one(x)["yes"] == pytest.approx(
            batch_model.predict_proba_many(x_batch)["yes"][0]
        )

        assert model.predict_proba_one(x)["no"] == pytest.approx(
            batch_model.predict_proba_many(x_batch)["no"][0]
        )

    # Assert class probabilities are the same when training Naive Bayes in pure online and in batch.
    assert model["model"].p_class("yes") == batch_model["model"].p_class("yes")
    assert model["model"].p_class("no") == batch_model["model"].p_class("no")

    # Assert conditional probabilities are the same when training Naive Bayes in pure online and
    # in batch.
    if isinstance(model["model"], naive_bayes.BernoulliNB) or isinstance(
        model["model"], naive_bayes.MultinomialNB
    ):
        inc_cp = model["model"].p_feature_given_class
        batch_cp = batch_model["model"].p_feature_given_class

        assert inc_cp("Chinese", "yes") == batch_cp("Chinese", "yes")
        assert inc_cp("Tokyo", "yes") == batch_cp("Tokyo", "yes")
        assert inc_cp("Japan", "yes") == batch_cp("Japan", "yes")
        assert inc_cp("Chinese", "no") == batch_cp("Chinese", "no")
        assert inc_cp("Tokyo", "no") == batch_cp("Tokyo", "no")
        assert inc_cp("Japan", "no") == batch_cp("Japan", "no")
        assert inc_cp("unseen", "yes") == batch_cp("unseen", "yes")

    assert model["model"].class_counts == batch_model["model"].class_counts
    assert model["model"].feature_counts == batch_model["model"].feature_counts

    if isinstance(model["model"], naive_bayes.ComplementNB) or isinstance(
        model["model"], naive_bayes.MultinomialNB
    ):
        assert model["model"].class_totals == batch_model["model"].class_totals

    if isinstance(model["model"], sk_naive_bayes.ComplementNB):
        assert model["model"].feature_totals == batch_model["model"].feature_totals


@pytest.mark.parametrize(
    "batch_model",
    [
        pytest.param(
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            id=f"{model.__name__} - {alpha}",
        )
        for model in river_models()
        for alpha in [alpha for alpha in range(1, 4)]
    ],
)
def test_learn_many_not_fit(batch_model):
    """Ensure that Naives Bayes models return an empty DataFrame when not yet fitted. Also check
    that the predict_proba_many method keeps the index.
    """
    assert batch_model.predict_proba_many(
        pd.Series(["new", "unseen"], index=["river", "rocks"])
    ).equals(pd.DataFrame(index=["river", "rocks"]))

    assert batch_model.predict_many(pd.Series(["new", "unseen"], index=["river", "rocks"])).equals(
        pd.DataFrame(index=["river", "rocks"])
    )


@pytest.mark.parametrize(
    "model, sk_model, bag",
    [
        pytest.param(
            compose.Pipeline(
                ("tokenize", feature_extraction.BagOfWords(lowercase=False)),
                ("model", model(alpha=alpha)),
            ),
            sk_model(alpha=alpha),
            feature_extraction.BagOfWords(lowercase=False),
            id=f"{model.__name__} - {alpha}",
        )
        for model, sk_model in sklearn_models()
        for alpha in [alpha for alpha in range(1, 4)]
    ],
)
def test_river_vs_sklearn(model, sk_model, bag):
    """Assert that river Naive Bayes models and sklearn Naive Bayes models provide the same results
    when the input data are the same. Also check that the behaviour of Naives Bayes models are the
    same with dense and sparse dataframe. Models tested are MultinomialNB, BernoulliNB and
    ComplementNB with different alpha parameters.
    """
    for x, y in yield_batch_dataset():
        model.learn_many(x, y)

    X = pd.concat([x for x, _ in yield_batch_dataset()])
    y = pd.concat([y for _, y in yield_batch_dataset()])

    sk_model = sk_model.fit(X=bag.transform_many(X), y=y)

    # Assert river produce same results as sklearn using sparse dataframe:
    for sk_preds, river_preds in zip(
        sk_model.predict_proba(bag.transform_many(X)),
        model.predict_proba_many(X).values,
    ):
        for sk_pred, river_pred in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(1 - sk_pred) or river_pred == pytest.approx(sk_pred)

    # Assert river produce same results as sklearn using dense dataframe:
    for sk_preds, river_preds in zip(
        sk_model.predict_proba(bag.transform_many(X).sparse.to_dense()),
        model["model"].predict_proba_many(bag.transform_many(X).sparse.to_dense()).values,
    ):
        for sk_pred, river_pred in zip(sk_preds, river_preds):
            assert river_pred == pytest.approx(1 - sk_pred) or river_pred == pytest.approx(sk_pred)


def test_gaussian_learn_many_vs_learn_one():
    X = pd.DataFrame(
        [
            {-1: -1.0, 1: -1.0},
            {-1: -2.0, 1: -1.0},
            {-1: -3.0},
            {-1: 1.0, 1: 1.0},
            {-1: 2.0, 1: 1.0},
            {-1: 3.0, 1: 2.0},
        ]
    )
    y = pd.Series([1, 1, 1, 2, 2, 2])

    model = naive_bayes.GaussianNB()
    batch_model = naive_bayes.GaussianNB()

    for _, row in X.iterrows():
        x = row.dropna().to_dict()
        model.learn_one(x, y.loc[row.name])

    batch_model.learn_many(X, y)

    assert batch_model.class_counts == model.class_counts
    assert batch_model.gaussians.keys() == model.gaussians.keys()

    for c, gaussians in model.gaussians.items():
        assert batch_model.gaussians[c].keys() == gaussians.keys()
        for i, gaussian in gaussians.items():
            batch_gaussian = batch_model.gaussians[c][i]
            assert batch_gaussian.n_samples == gaussian.n_samples
            assert batch_gaussian.mu == pytest.approx(gaussian.mu)
            assert batch_gaussian.sigma == pytest.approx(gaussian.sigma)

    X_unseen = pd.DataFrame(
        [
            {-1: -0.8, 1: -1.0},
            {-1: 2.8, 1: 1.5},
            {1: -1.0},
            {-1: 4.0, 2: 10.0},
        ],
        index=["a", "b", "c", "d"],
    )

    batch_proba = batch_model.predict_proba_many(X_unseen)

    assert batch_proba.index.tolist() == X_unseen.index.tolist()
    assert batch_proba.columns.tolist() == [1, 2]
    assert batch_model.predict_many(X_unseen).tolist() == [
        batch_model.predict_one(row.dropna().to_dict()) for _, row in X_unseen.iterrows()
    ]

    for i, row in X_unseen.iterrows():
        proba_one = model.predict_proba_one(row.dropna().to_dict())
        assert batch_proba.loc[i, 1] == pytest.approx(proba_one[1])
        assert batch_proba.loc[i, 2] == pytest.approx(proba_one[2])


def test_gaussian_learn_many_sparse():
    X = pd.DataFrame({0: [-1.0, -2.0, 1.0, 2.0], 1: [-1.0, -1.0, 1.0, 1.0]}).astype(
        pd.SparseDtype(float, 0.0)
    )
    y = pd.Series(["a", "a", "b", "b"])

    model = naive_bayes.GaussianNB()
    batch_model = naive_bayes.GaussianNB()

    for x, yi in zip(X.sparse.to_dense().to_dict(orient="records"), y):
        model.learn_one(x, yi)

    batch_model.learn_many(X, y)

    assert batch_model.predict_proba_many(X).equals(model.predict_proba_many(X.sparse.to_dense()))


def test_gaussian_learn_many_sklearn_partial_fit():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        np.vstack(
            [
                rng.normal(loc=[-1.0, 0.0], scale=[1.2, 0.8], size=(200, 2)),
                rng.normal(loc=[1.0, 0.5], scale=[1.2, 0.8], size=(200, 2)),
            ]
        )
    )
    y = pd.Series(["a"] * 200 + ["b"] * 200)
    classes = np.array(["a", "b"])
    order = rng.permutation(len(X))
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    X_test = pd.DataFrame([[-1.0, 0.0], [1.0, 0.5], [0.0, 0.25], [-0.2, 0.1], [0.2, 0.3]])

    model = naive_bayes.GaussianNB()
    sk_model = sk_naive_bayes.GaussianNB(var_smoothing=0.0)

    for start in range(0, len(X), 50):
        X_batch = X.iloc[start : start + 50]
        y_batch = y.iloc[start : start + 50]
        model.learn_many(X_batch, y_batch)
        sk_model.partial_fit(X_batch, y_batch, classes=classes)

        river_means = np.array([[model.gaussians[c][i].mu for i in X.columns] for c in classes])
        river_priors = np.array([model.p_class(c) for c in classes])

        np.testing.assert_allclose(river_means, sk_model.theta_)
        np.testing.assert_allclose(river_priors, sk_model.class_prior_)
        assert model.predict_many(X_test).tolist() == sk_model.predict(X_test).tolist()
        np.testing.assert_allclose(
            model.predict_proba_many(X_test)[classes].to_numpy(),
            sk_model.predict_proba(X_test),
            atol=2e-2,
        )


def test_gaussian_learn_many_not_fit():
    model = naive_bayes.GaussianNB()
    X = pd.DataFrame([{0: 1.0}, {0: 2.0}], index=["river", "rocks"])

    assert model.predict_proba_many(X).equals(pd.DataFrame(index=["river", "rocks"]))
    assert model.predict_many(X).equals(pd.DataFrame(index=["river", "rocks"]))


def _categorical_dataset(seed, n=80):
    import random

    rng = random.Random(seed)
    weather = ["sunny", "overcast", "rainy"]
    humidity = ["high", "normal"]
    rows, ys = [], []
    for _ in range(n):
        w, h = rng.choice(weather), rng.choice(humidity)
        rows.append({"weather": w, "humidity": h})
        ys.append("yes" if (w == "overcast" or h == "normal") else "no")
    return weather, humidity, rows, ys


@pytest.mark.parametrize("alpha", [1.0, 2.0, 3.0])
def test_categorical_vs_sklearn(alpha):
    """river's CategoricalNB must match sklearn's CategoricalNB on categorical data."""
    weather, humidity, rows, ys = _categorical_dataset(seed=42)
    wmap = {c: i for i, c in enumerate(weather)}
    hmap = {c: i for i, c in enumerate(humidity)}
    ymap = {"no": 0, "yes": 1}
    inv_y = {v: k for k, v in ymap.items()}
    X = np.array([[wmap[r["weather"]], hmap[r["humidity"]]] for r in rows])
    Y = np.array([ymap[y] for y in ys])

    river_model = naive_bayes.CategoricalNB(alpha=alpha)
    for r, y in zip(rows, ys):
        river_model.learn_one(r, y)

    sk = sk_naive_bayes.CategoricalNB(alpha=alpha).fit(X, Y)

    for r, xrow in zip(rows, X):
        river_proba = river_model.predict_proba_one(r)
        sk_proba = sk.predict_proba(xrow.reshape(1, -1))[0]
        for idx, cls in enumerate(sk.classes_):
            assert river_proba[inv_y[cls]] == pytest.approx(sk_proba[idx])


def test_categorical_learn_many_vs_learn_one():
    """CategoricalNB.learn_many must yield the same model as repeated learn_one."""
    _, _, rows, ys = _categorical_dataset(seed=7, n=40)

    one = naive_bayes.CategoricalNB(alpha=1)
    for r, y in zip(rows, ys):
        one.learn_one(r, y)

    many = naive_bayes.CategoricalNB(alpha=1)
    many.learn_many(pd.DataFrame(rows), pd.Series(ys))

    assert one.class_counts == many.class_counts
    assert one.feature_counts == many.feature_counts

    test_x = {"weather": "overcast", "humidity": "normal"}
    assert one.predict_proba_one(test_x) == pytest.approx(many.predict_proba_one(test_x))


def test_categorical_handles_unseen_feature_value():
    """An unseen category at predict time must not raise and must stay normalized."""
    model = naive_bayes.CategoricalNB(alpha=1)
    for x, y in [
        ({"weather": "sunny"}, "no"),
        ({"weather": "rainy"}, "yes"),
    ]:
        model.learn_one(x, y)

    proba = model.predict_proba_one({"weather": "snowy"})  # category never seen
    assert set(proba) == {"no", "yes"}
    assert sum(proba.values()) == pytest.approx(1.0)
