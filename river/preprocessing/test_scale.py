from __future__ import annotations

import math

import numpy as np
import pandas as pd

from river import datasets, preprocessing, stream


def test_standard_scaler_one_many_consistent():
    """Checks that using learn_one or learn_many produces the same result."""

    for with_std in (False, True):
        X = pd.read_csv(datasets.TrumpApproval().path)

        one = preprocessing.StandardScaler(with_std=with_std)
        for x, _ in stream.iter_pandas(X):
            one.learn_one(x)

        many = preprocessing.StandardScaler(with_std=with_std)
        for xb in np.array_split(X, 10):
            many.learn_many(xb)

        for i in X:
            assert math.isclose(one.counts[i], many.counts[i])
            assert math.isclose(one.means[i], many.means[i])
            assert math.isclose(one.vars[i], many.vars[i])


def test_standard_scaler_shuffle_columns():
    """Checks that learn_many works identically whether columns are shuffled or not."""

    X = pd.read_csv(datasets.TrumpApproval().path)

    normal = preprocessing.StandardScaler()
    for xb in np.array_split(X, 10):
        normal.learn_many(xb)

    shuffled = preprocessing.StandardScaler()
    for xb in np.array_split(X, 10):
        cols = np.random.permutation(X.columns)
        shuffled.learn_many(xb[cols])

    for i in X:
        assert math.isclose(shuffled.counts[i], shuffled.counts[i])
        assert math.isclose(shuffled.means[i], shuffled.means[i])
        assert math.isclose(shuffled.vars[i], shuffled.vars[i])


def test_standard_scaler_add_remove_columns():
    """Checks that no exceptions are raised whenever columns are dropped and/or added."""

    X = pd.read_csv(datasets.TrumpApproval().path)

    ss = preprocessing.StandardScaler()
    for xb in np.array_split(X, 10):
        # Pick half of the columns at random
        cols = np.random.choice(X.columns, len(X.columns) // 2, replace=False)
        ss.learn_many(xb[cols])


def test_issue_1313():
    """

    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from river import preprocessing
    >>> from river.compose import Select

    >>> X, y = datasets.make_regression(n_samples=6, n_features=2)
    >>> X = pd.DataFrame(X)
    >>> X.columns = ['feat_1','feat_2']

    >>> model = Select('feat_1') | preprocessing.StandardScaler()
    >>> X = X.astype('float32')
    >>> X.dtypes
    feat_1    float32
    feat_2    float32
    dtype: object

    >>> model = model.learn_many(X)
    >>> X1 = model.transform_many(X)
    >>> X1.dtypes
    feat_1    float32
    dtype: object

    """
