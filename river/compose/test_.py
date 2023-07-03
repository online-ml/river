from __future__ import annotations

import pandas as pd

from river import compose, linear_model, preprocessing


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


def test_learn_one_with_learn_during_predict():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), bool(x % 2)) for x in range(100)]

    for x, y in dataset:
        counts_pre = dict(pipeline["scale"].counts)
        with compose.learn_during_predict():
            pipeline.learn_one(x, y)
        counts_no_learn = dict(pipeline["scale"].counts)
        pipeline.learn_one(x, y)
        counts_post = dict(pipeline["scale"].counts)

        assert counts_pre != counts_post
        assert counts_pre == counts_no_learn


def test_learn_many_with_learn_during_predict():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("log_reg", linear_model.LogisticRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for i in range(0, len(dataset), 5):
        X = pd.DataFrame([x for x, _ in dataset][i : i + 5])
        y = pd.Series([bool(y % 2) for _, y in dataset][i : i + 5])

        counts_pre = dict(pipeline["scale"].counts)
        with compose.learn_during_predict():
            pipeline.learn_many(X, y)
        counts_no_learn = dict(pipeline["scale"].counts)
        pipeline.learn_many(X, y)
        counts_post = dict(pipeline["scale"].counts)

        assert counts_pre != counts_post
        assert counts_pre == counts_no_learn


def test_list_of_funcs():
    def f(x):
        return {"f": 1}

    def g(x):
        return {"g": 2}

    def times_2(x):
        return {k: v * 2 for k, v in x.items()}

    expected = {"f": 2, "g": 4}
    assert compose.Pipeline([f, g], times_2).transform_one(None) == expected
    assert ([f, g] | compose.FuncTransformer(times_2)).transform_one(None) == expected


def test_get():
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()

    assert isinstance(model["StandardScaler"], preprocessing.StandardScaler)
    assert isinstance(model["LinearRegression"], linear_model.LinearRegression)
    assert isinstance(model[0], preprocessing.StandardScaler)
    assert isinstance(model[1], linear_model.LinearRegression)
    assert isinstance(model[-1], linear_model.LinearRegression)
    assert isinstance(model[-2], preprocessing.StandardScaler)
