from river import compose, preprocessing


def test_left_is_pipeline():

    group_1 = compose.Select("a", "b")
    group_2 = compose.Select("x", "y") | preprocessing.OneHotEncoder(sparse=True)

    product = group_1 + group_2 + group_1 * group_2
    assert product.transform_one(dict(a=1, b=2, x=4, y=4, z=5)) == {
        "a*y_4": 1,
        "a*x_4": 1,
        "b*y_4": 2,
        "b*x_4": 2,
        "y_4": 1,
        "x_4": 1,
        "a": 1,
        "b": 2,
    }


def test_right_is_pipeline():

    group_1 = compose.Select("a", "b") | preprocessing.OneHotEncoder(sparse=True)
    group_2 = compose.Select("x", "y")

    product = group_1 + group_2 + group_1 * group_2
    assert product.transform_one(dict(a=1, b=2, x=4, y=4, z=5)) == {
        "a_1*x": 4,
        "a_1*y": 4,
        "b_2*x": 4,
        "b_2*y": 4,
        "x": 4,
        "y": 4,
        "a_1": 1,
        "b_2": 1,
    }


def test_both_are_pipelines():

    group_1 = compose.Select("a", "b") | preprocessing.OneHotEncoder(sparse=True)
    group_2 = compose.Select("x", "y") | preprocessing.OneHotEncoder(sparse=True)

    product = group_1 + group_2 + group_1 * group_2
    assert product.transform_one(dict(a=1, b=2, x=4, y=4, z=5)) == {
        "b_2*x_4": 1,
        "b_2*y_4": 1,
        "a_1*x_4": 1,
        "a_1*y_4": 1,
        "x_4": 1,
        "y_4": 1,
        "b_2": 1,
        "a_1": 1,
    }


def test_renaming():

    renamer = compose.Renamer(dict(a="z", b="y", c="x"))
    assert renamer.transform_one(dict(a=1, b=2, d=3)) == dict(z=1, y=2, d=3)


def test_prefixing():

    prefixer = compose.Prefixer("x_")
    assert prefixer.transform_one(dict(a=1, b=2, d=3)) == dict(x_a=1, x_b=2, x_d=3)


def test_suffixing():

    suffixer = compose.Suffixer("_x")
    assert suffixer.transform_one(dict(a=1, b=2, d=3)) == dict(a_x=1, b_x=2, d_x=3)
