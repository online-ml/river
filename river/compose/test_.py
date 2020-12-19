from river import compose
from river import linear_model
from river import preprocessing


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
