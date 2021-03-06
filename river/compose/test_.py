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


def test_no_learn_predict_one():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("lin_reg", linear_model.LinearRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for x, y in dataset:
        counts_pre = dict(pipeline.steps["scale"].counts)
        pipeline.predict_one(x, no_learn=False)
        counts_post = dict(pipeline.steps["scale"].counts)
        pipeline.predict_one(x, no_learn=True)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn


def test_no_learn_predict_many():
    pipeline = compose.Pipeline(
        ("scale", preprocessing.StandardScaler()),
        ("lin_reg", linear_model.LinearRegression()),
    )

    dataset = [(dict(a=x, b=x), x) for x in range(100)]

    for i in range(0, len(dataset), 5):
        X = pd.DataFrame([x for x, y in dataset][i : i + 5])

        counts_pre = dict(pipeline.steps["scale"].counts)
        pipeline.predict_many(X, no_learn=False)
        counts_post = dict(pipeline.steps["scale"].counts)
        pipeline.predict_many(X, no_learn=True)
        counts_no_learn = dict(pipeline.steps["scale"].counts)

        assert counts_pre != counts_post
        assert counts_post == counts_no_learn
