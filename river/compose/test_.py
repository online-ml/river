import pandas as pd
import pytest

from river import anomaly, compose, linear_model, preprocessing


def test_pipeline_funcs():
    def a(x):
        pass

    def b(x):
        pass

    pipelines = [
        compose.FuncTransformer(a) | b,
        compose.FuncTransformer(a) | ("b", b),
        compose.FuncTransformer(a) | ("b", compose.FuncTransformer(b)),
        a | compose.FuncTransformer(b),
        ("a", a) | compose.FuncTransformer(b),
        ("a", compose.FuncTransformer(a)) | compose.FuncTransformer(b),
    ]

    for pipeline in pipelines:
        assert str(pipeline) == "a | b"


def test_pipeline_add_at_start():
    def a(x):
        pass

    pipeline = preprocessing.StandardScaler() | linear_model.LinearRegression()
    pipeline = a | pipeline
    assert str(pipeline) == "a | StandardScaler | LinearRegression"


def test_union_funcs():
    def a(x):
        pass

    def b(x):
        pass

    pipelines = [
        compose.FuncTransformer(a) + b,
        compose.FuncTransformer(a) + ("b", b),
        compose.FuncTransformer(a) + ("b", compose.FuncTransformer(b)),
        a + compose.FuncTransformer(b),
        ("a", a) + compose.FuncTransformer(b),
        ("a", compose.FuncTransformer(a)) + compose.FuncTransformer(b),
    ]

    for i, pipeline in enumerate(pipelines):
        print(i, str(pipeline))
        assert str(pipeline) == "a + b"


@pytest.mark.parametrize(
    "func", [compose.Pipeline.predict_one, compose.Pipeline.predict_proba_one]
)
def test_learn_unsupervised_predict_one(func):
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [dict(a=x, b=x) for x in range(100)]

    for x in dataset:
        counts_pre = dict(pipeline.steps["scale"].counts)
        func(pipeline, x, learn_unsupervised=True)
        counts_post = dict(pipeline.steps["scale"].counts)
        func(pipeline, x, learn_unsupervised=False)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn


@pytest.mark.parametrize(
    "func", [compose.Pipeline.predict_many, compose.Pipeline.predict_proba_many]
)
def test_learn_unsupervised_predict_many(func):
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for i in range(0, len(dataset), 5):
        X = pd.DataFrame([x for x, y in dataset][i : i + 5])

        counts_pre = dict(pipeline.steps["scale"].counts)
        func(pipeline, X, learn_unsupervised=True)
        counts_post = dict(pipeline.steps["scale"].counts)
        func(pipeline, X, learn_unsupervised=False)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn


def test_learn_unsupervised_score_one():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("anomaly", anomaly.HalfSpaceTrees()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for x, y in dataset:
        counts_pre = dict(pipeline.steps["scale"].counts)
        pipeline.score_one(x, learn_unsupervised=True)
        counts_post = dict(pipeline.steps["scale"].counts)
        pipeline.score_one(x, learn_unsupervised=False)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn


def test_learn_unsupervised_learn_one():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), bool(x % 2)) for x in range(100)]

    for x, y in dataset:
        counts_pre = dict(pipeline.steps["scale"].counts)
        pipeline.learn_one(x, y, learn_unsupervised=True)
        counts_post = dict(pipeline.steps["scale"].counts)
        pipeline.learn_one(x, y, learn_unsupervised=False)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn


def test_learn_unsupervised_learn_many():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for i in range(0, len(dataset), 5):
        X = pd.DataFrame([x for x, _ in dataset][i : i + 5])
        y = pd.Series([bool(y % 2) for _, y in dataset][i : i + 5])

        counts_pre = dict(pipeline.steps["scale"].counts)
        pipeline.learn_many(X, y, learn_unsupervised=True)
        counts_post = dict(pipeline.steps["scale"].counts)
        pipeline.learn_many(X, y, learn_unsupervised=False)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn
