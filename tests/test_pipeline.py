from creme import compose


def test_union():

    def a(x):
        pass

    def b(x):
        pass

    pipelines = [
        compose.FuncTransformer(a) | b,
        compose.FuncTransformer(a) | ('b', b),
        compose.FuncTransformer(a) | ('b', compose.FuncTransformer(b)),
        a | compose.FuncTransformer(b),
        ('a', a) | compose.FuncTransformer(b),
        ('a', compose.FuncTransformer(a)) | compose.FuncTransformer(b)
    ]

    for pipeline in pipelines:
        assert str(pipeline) == 'a | b'
