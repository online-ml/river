from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import datasets as sk_datasets
from sklearn import linear_model as sk_linear_model

from river import compat, compose, datasets, preprocessing, stream


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
    {'z*x': 0.785..., 'x': 0.785..., 'z': 1}
    {'z*x': -1.511..., 'x': -1.511..., 'z': 1}
    {'z*x': -0.576..., 'x': -0.576..., 'z': 1}
    {'z*x': -0.770..., 'x': -0.770..., 'z': 1}
    {'z*x': 1.148..., 'x': 1.148..., 'z': 1}
    {'z*x': 0.924..., 'x': 0.924..., 'z': 1}

    """


def test_issue_1253():
    """

    https://github.com/online-ml/river/issues/1253

    >>> import numpy as np
    >>> import pandas as pd
    >>> from river import compat, compose, preprocessing
    >>> from sklearn import datasets, linear_model

    >>> np.random.seed(1000)
    >>> X, y = datasets.make_regression(n_samples=5_000, n_features=2)
    >>> X = pd.DataFrame(X, columns=['feat_1','feat_2'])
    >>> X['cat'] = np.random.randint(1, 100, len(X))
    >>> X['cat'] = X['cat'].astype('string')

    >>> group1 = compose.Select('cat') | preprocessing.OneHotEncoder()
    >>> group2 = compose.Select('feat_2') | preprocessing.StandardScaler()
    >>> model = group1 + group1 * group2
    >>> XT = model.transform_many(X)

    >>> XT.memory_usage().sum() // 1000
    85

    >>> XT.sparse.to_dense().memory_usage().sum() // 1000
    4455

    >>> X, y = datasets.make_regression(n_samples=6, n_features=2)
    >>> X = pd.DataFrame(X)
    >>> X.columns = ['feat_1','feat_2']
    >>> X['cat'] = np.random.randint(1, 3, X.shape[0])
    >>> y = pd.Series(y)
    >>> group1 = compose.Select('cat') | preprocessing.OneHotEncoder()
    >>> group2 = compose.Select('feat_2') | preprocessing.StandardScaler()
    >>> sparsify = lambda X: X.astype({
    ...     key: pd.SparseDtype(X.dtypes[key].type, fill_value=0)
    ...     for key in X.dtypes.keys()
    ... })
    >>> model = (
    ...     (group1 + group1 * group2) |
    ...     compose.FuncTransformer(sparsify) |
    ...     compat.convert_sklearn_to_river(linear_model.SGDRegressor(max_iter=3))
    ... )
    >>> _ = model.predict_many(X)
    >>> model.transform_many(X)
       cat_1*feat_2  cat_2*feat_2  cat_1  cat_2
    0     -1.196841      0.000000      1      0
    1      1.304619      0.000000      1      0
    2     -1.294091      0.000000      1      0
    3      0.287426      0.000000      1      0
    4     -0.143960      0.000000      1      0
    5      0.000000      1.042847      0      1

    """


def test_left_is_pipeline():
    group_1 = compose.Select("a", "b")
    group_2 = compose.Select("x", "y") | preprocessing.OneHotEncoder()

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
    group_1 = compose.Select("a", "b") | preprocessing.OneHotEncoder()
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
    group_1 = compose.Select("a", "b") | preprocessing.OneHotEncoder()
    group_2 = compose.Select("x", "y") | preprocessing.OneHotEncoder()

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


def test_issue_1310():
    X, y = sk_datasets.make_regression(n_samples=5000, n_features=2)
    X = pd.DataFrame(X)
    X.columns = ["feat_1", "feat_2"]
    X["cat"] = np.random.randint(1, 100, X.shape[0])
    X["cat"] = X["cat"].astype("string")
    y = pd.Series(y)

    group1 = compose.Select("cat") | preprocessing.OneHotEncoder()
    group2 = compose.Select("feat_2") | preprocessing.StandardScaler()
    model = group1 + group1 * group2 * group2 | compat.convert_sklearn_to_river(
        sk_linear_model.SGDRegressor()
    )

    model.predict_many(X)
    model.learn_many(X, y)
