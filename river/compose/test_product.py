from __future__ import annotations

import pandas as pd

from river import compose, datasets, preprocessing, stream


def test_issue_1238():
    """

    https://github.com/online-ml/river/issues/1238

    >>> from river import compose

    >>> x = dict(a=0, b=1, x=2, y=3)
    >>> group_1 = compose.Select('a', 'b')
    >>> group_2 = compose.Select('x', 'y')
    >>> product = group_1 + group_2 + group_1 * group_2
    >>> sorted(product.transform_one(x), key=lambda k: (len(k), k))
    ['a', 'b', 'x', 'y', 'a*x', 'a*y', 'b*x', 'b*y']

    >>> product = group_1 * group_2 + group_1 + group_2
    >>> sorted(product.transform_one(x), key=lambda k: (len(k), k))
    ['a', 'b', 'x', 'y', 'a*x', 'a*y', 'b*x', 'b*y']

    >>> product = group_1 + group_1 * group_2
    >>> sorted(product.transform_one(x), key=lambda k: (len(k), k))
    ['a', 'b', 'a*x', 'a*y', 'b*x', 'b*y']

    """


def test_issue_1243():
    """

    https://github.com/online-ml/river/issues/1243

    >>> import random
    >>> import pandas as pd
    >>> from river import compose, preprocessing

    >>> rng = random.Random(42)
    >>> X = [{'x': rng.uniform(8, 12), 'z': 1 } for _ in range(6)]

    >>> X = pd.DataFrame.from_dict(X)
    >>> group1 = compose.Select('z')
    >>> group2 = compose.Select('x') | preprocessing.StandardScaler()
    >>> model = group1 + group2 + group1 * group2
    >>> model = model.learn_many(X)
    >>> for x in X.to_dict('records'):
    ...     print(model.transform_one(x))
    {'z*x': 0.697074..., 'x': 0.697074..., 'z': 1}
    {'z*x': -1.351716..., 'x': -1.351716..., 'z': 1}
    {'z*x': -0.430886..., 'x': -0.430886..., 'z': 1}
    {'z*x': -0.580868..., 'x': -0.580868..., 'z': 1}
    {'z*x': 1.228463..., 'x': 1.228463..., 'z': 1}
    {'z*x': 0.924721..., 'x': 0.924721..., 'z': 1}

    """


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


def test_one_many_consistent():
    """Checks that using transform_one or transform_many produces the same result."""

    product = (
        compose.Select("ordinal_date")
        + compose.Select("gallup", "ipsos") * compose.Select("morning_consult")
        + compose.Select("rasmussen") * compose.Select("you_gov", "five_thirty_eight")
    )
    X = pd.read_csv(datasets.TrumpApproval().path)

    one_outputs = []
    for x, _ in stream.iter_pandas(X):
        one_outputs.append(product.transform_one(x))
    one_outputs = pd.DataFrame(one_outputs)

    many_outputs = product.transform_many(X)

    # check_dtype=False to avoid int/float comparison
    pd.testing.assert_frame_equal(many_outputs[one_outputs.columns], one_outputs, check_dtype=False)
