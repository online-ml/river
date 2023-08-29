from __future__ import annotations

import collections
import functools
import itertools
import numbers

import numpy as np
import pandas as pd

from river import base, stats, utils

__all__ = [
    "AdaptiveStandardScaler",
    "Binarizer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "RobustScaler",
    "StandardScaler",
]


def safe_div(a, b):
    """Returns a if b is nil, else divides a by b.

    When scaling, sometimes a denominator might be nil. For instance, during standard scaling
    the denominator can be nil if a feature has no variance.

    """
    return a / b if b else 0.0


class Binarizer(base.Transformer):
    """Binarizes the data to 0 or 1 according to a threshold.

    Parameters
    ----------
    threshold
        Values above this are replaced by 1 and the others by 0.
    dtype
        The desired data type to apply.

    Examples
    --------

    >>> import river
    >>> import numpy as np

    >>> rng = np.random.RandomState(42)
    >>> X = [{'x1': v, 'x2': int(v)} for v in rng.uniform(low=-4, high=4, size=6)]

    >>> binarizer = river.preprocessing.Binarizer()
    >>> for x in X:
    ...     print(binarizer.learn_one(x).transform_one(x))
    {'x1': False, 'x2': False}
    {'x1': True, 'x2': True}
    {'x1': True, 'x2': True}
    {'x1': True, 'x2': False}
    {'x1': False, 'x2': False}
    {'x1': False, 'x2': False}

    """

    def __init__(self, threshold=0.0, dtype=bool):
        self.threshold = threshold
        self.dtype = dtype

    def transform_one(self, x):
        x_tf = x.copy()

        for i, xi in x_tf.items():
            if isinstance(xi, numbers.Number):
                x_tf[i] = self.dtype(xi > self.threshold)

        return x_tf


class StandardScaler(base.MiniBatchTransformer):
    """Scales the data so that it has zero mean and unit variance.

    Under the hood, a running mean and a running variance are maintained. The scaling is slightly
    different than when scaling the data in batch because the exact means and variances are not
    known in advance. However, this doesn't have a detrimental impact on performance in the long
    run.

    This transformer supports mini-batches as well as single instances. In the mini-batch case, the
    number of columns and the ordering of the columns are allowed to change between subsequent
    calls. In other words, this transformer will keep working even if you add and/or remove
    features every time you call `learn_many` and `transform_many`.

    Parameters
    ----------
    with_std
        Whether or not each feature should be divided by its standard deviation.

    Examples
    --------

    >>> import random
    >>> from river import preprocessing

    >>> random.seed(42)
    >>> X = [{'x': random.uniform(8, 12), 'y': random.uniform(8, 12)} for _ in range(6)]
    >>> for x in X:
    ...     print(x)
    {'x': 10.557, 'y': 8.100}
    {'x': 9.100, 'y': 8.892}
    {'x': 10.945, 'y': 10.706}
    {'x': 11.568, 'y': 8.347}
    {'x': 9.687, 'y': 8.119}
    {'x': 8.874, 'y': 10.021}

    >>> scaler = preprocessing.StandardScaler()

    >>> for x in X:
    ...     print(scaler.learn_one(x).transform_one(x))
    {'x': 0.0, 'y': 0.0}
    {'x': -0.999, 'y': 0.999}
    {'x': 0.937, 'y': 1.350}
    {'x': 1.129, 'y': -0.651}
    {'x': -0.776, 'y': -0.729}
    {'x': -1.274, 'y': 0.992}

    This transformer also supports mini-batch updates. You can call `learn_many` and provide a
    `pandas.DataFrame`:

    >>> import pandas as pd
    >>> X = pd.DataFrame.from_dict(X)

    >>> scaler = preprocessing.StandardScaler()
    >>> scaler = scaler.learn_many(X[:3])
    >>> scaler = scaler.learn_many(X[3:])

    You can then call `transform_many` to scale a mini-batch of features:

    >>> scaler.transform_many(X)
        x         y
    0  0.444600 -0.933384
    1 -1.044259 -0.138809
    2  0.841106  1.679208
    3  1.477301 -0.685117
    4 -0.444084 -0.914195
    5 -1.274664  0.992296

    References
    ----------
    [^1]: [Welford's Method (and Friends)](https://www.embeddedrelated.com/showarticle/785.php)
    [^2]: [Batch updates for simple statistics](https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)

    """

    def __init__(self, with_std=True):
        self.with_std = with_std
        self.counts = collections.Counter()
        self.means = collections.defaultdict(float)
        self.vars = collections.defaultdict(float)

    def learn_one(self, x):
        for i, xi in x.items():
            self.counts[i] += 1
            old_mean = self.means[i]
            self.means[i] += (xi - old_mean) / self.counts[i]
            if self.with_std:
                self.vars[i] += (
                    (xi - old_mean) * (xi - self.means[i]) - self.vars[i]
                ) / self.counts[i]

        return self

    def transform_one(self, x):
        if self.with_std:
            return {i: safe_div(xi - self.means[i], self.vars[i] ** 0.5) for i, xi in x.items()}
        return {i: xi - self.means[i] for i, xi in x.items()}

    def learn_many(self, X: pd.DataFrame):
        """Update with a mini-batch of features.

        Note that the update formulas for mean and variance are slightly different than in the
        single instance case, but they produce exactly the same result.

        Parameters
        ----------
        X
            A dataframe where each column is a feature.

        """

        # Operating on X.values, which is a view to the underlying numpy array, is slightly faster
        # than operating on X
        columns = X.columns
        X = X.values

        # In the rest of this method, old_* refers to the existing statistics, whilst new_* refers
        # to the statistics of the current mini-batch.

        new_means = np.nanmean(X, axis=0)
        # We could call np.var, but we already have the mean so we can be smart
        if self.with_std:
            new_vars = np.einsum("ij,ij->j", X, X) / len(X) - new_means**2
        else:
            new_vars = []
        new_counts = np.sum(~np.isnan(X), axis=0)

        for col, new_mean, new_var, new_count in itertools.zip_longest(
            columns, new_means, new_vars, new_counts
        ):
            old_mean = self.means[col]
            old_var = self.vars[col]
            old_count = self.counts[col]

            a = old_count / (old_count + new_count)
            b = new_count / (old_count + new_count)

            self.means[col] = a * old_mean + b * new_mean
            if self.with_std:
                self.vars[col] = a * old_var + b * new_var + a * b * (old_mean - new_mean) ** 2
            self.counts[col] += new_count

        return self

    def transform_many(self, X: pd.DataFrame):
        """Scale a mini-batch of features.

        Parameters
        ----------
        X
            A dataframe where each column is a feature. An exception will be raised if any of
            the features has not been seen during a previous call to `learn_many`.

        """

        # Determine dtype of input
        dtypes = X.dtypes.unique()
        dtype = dtypes[0] if len(dtypes) == 1 else np.float64

        # Check if the dtype is integer type and convert to corresponding float type
        if np.issubdtype(dtype, np.integer):
            bytes_size = dtype.itemsize
            dtype = np.dtype(f"float{bytes_size * 8}")

        means = np.array([self.means[c] for c in X.columns], dtype=dtype)
        Xt = X.values - means

        if self.with_std:
            stds = np.array([self.vars[c] ** 0.5 for c in X.columns], dtype=dtype)
            np.divide(Xt, stds, where=stds > 0, out=Xt)

        return pd.DataFrame(Xt, index=X.index, columns=X.columns, copy=False)


class MinMaxScaler(base.Transformer):
    """Scales the data to a fixed range from 0 to 1.

    Under the hood a running min and a running peak to peak (max - min) are maintained.

    Attributes
    ----------
    min : dict
        Mapping between features and instances of `stats.Min`.
    max : dict
        Mapping between features and instances of `stats.Max`.

    Examples
    --------

    >>> import random
    >>> from river import preprocessing

    >>> random.seed(42)
    >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
    >>> for x in X:
    ...     print(x)
    {'x': 10.557707}
    {'x': 8.100043}
    {'x': 9.100117}
    {'x': 8.892842}
    {'x': 10.945884}

    >>> scaler = preprocessing.MinMaxScaler()

    >>> for x in X:
    ...     print(scaler.learn_one(x).transform_one(x))
    {'x': 0.0}
    {'x': 0.0}
    {'x': 0.406920}
    {'x': 0.322582}
    {'x': 1.0}

    """

    def __init__(self):
        self.min = collections.defaultdict(stats.Min)
        self.max = collections.defaultdict(stats.Max)

    def learn_one(self, x):
        for i, xi in x.items():
            self.min[i].update(xi)
            self.max[i].update(xi)

        return self

    def transform_one(self, x):
        return {
            i: safe_div(xi - self.min[i].get(), self.max[i].get() - self.min[i].get())
            for i, xi in x.items()
        }


class MaxAbsScaler(base.Transformer):
    """Scales the data to a [-1, 1] range based on absolute maximum.

    Under the hood a running absolute max is maintained. This scaler is meant for
    data that is already centered at zero or sparse data. It does not shift/center
    the data, and thus does not destroy any sparsity.

    Attributes
    ----------
    abs_max : dict
        Mapping between features and instances of `stats.AbsMax`.

    Examples
    --------

    >>> import random
    >>> from river import preprocessing

    >>> random.seed(42)
    >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
    >>> for x in X:
    ...     print(x)
    {'x': 10.557707}
    {'x': 8.100043}
    {'x': 9.100117}
    {'x': 8.892842}
    {'x': 10.945884}

    >>> scaler = preprocessing.MaxAbsScaler()

    >>> for x in X:
    ...     print(scaler.learn_one(x).transform_one(x))
    {'x': 1.0}
    {'x': 0.767216}
    {'x': 0.861940}
    {'x': 0.842308}
    {'x': 1.0}

    """

    def __init__(self):
        self.abs_max = collections.defaultdict(stats.AbsMax)

    def learn_one(self, x):
        for i, xi in x.items():
            self.abs_max[i].update(xi)

        return self

    def transform_one(self, x):
        return {i: safe_div(xi, self.abs_max[i].get()) for i, xi in x.items()}


class RobustScaler(base.Transformer):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the
    interquantile range.

    Parameters
    ----------
    with_centering
        Whether to centre the data before scaling.
    with_scaling
        Whether to scale data to IQR.
    q_inf
        Desired inferior quantile, must be between 0 and 1.
    q_sup
        Desired superior quantile, must be between 0 and 1.

    Attributes
    ----------
    median : dict
        Mapping between features and instances of `stats.Quantile(0.5)`.
    iqr : dict
        Mapping between features and instances of `stats.IQR`.

    Examples
    --------

    >>> from pprint import pprint
    >>> import random
    >>> from river import preprocessing

    >>> random.seed(42)
    >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
    >>> pprint(X)
    [{'x': 10.557707},
        {'x': 8.100043},
        {'x': 9.100117},
        {'x': 8.892842},
        {'x': 10.945884}]

    >>> scaler = preprocessing.RobustScaler()

    >>> for x in X:
    ...     print(scaler.learn_one(x).transform_one(x))
        {'x': 0.0}
        {'x': -1.0}
        {'x': 0.0}
        {'x': -0.12449923287875722}
        {'x': 1.1086595155704708}

    """

    def __init__(self, with_centering=True, with_scaling=True, q_inf=0.25, q_sup=0.75):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.median = collections.defaultdict(functools.partial(stats.Quantile, 0.5))
        self.iqr = collections.defaultdict(functools.partial(stats.IQR, self.q_inf, self.q_sup))

    def learn_one(self, x):
        for i, xi in x.items():
            if self.with_centering:
                self.median[i].update(xi)
            if self.with_scaling:
                self.iqr[i].update(xi)

        return self

    def transform_one(self, x):
        x_tf = {}

        for i, xi in x.items():
            x_tf[i] = xi
            if self.with_centering:
                x_tf[i] -= self.median[i].get()
            if self.with_scaling:
                x_tf[i] = safe_div(x_tf[i], self.iqr[i].get())

        return x_tf


class Normalizer(base.Transformer):
    """Scales a set of features so that it has unit norm.

    This is particularly useful when used after a `feature_extraction.TFIDF`.

    Parameters
    ----------
    order
        Order of the norm (e.g. 2 corresponds to the $L^2$ norm).

    Examples
    --------

    >>> from river import preprocessing
    >>> from river import stream

    >>> scaler = preprocessing.Normalizer(order=2)

    >>> X = [[4, 1, 2, 2],
    ...      [1, 3, 9, 3],
    ...      [5, 7, 5, 1]]

    >>> for x, _ in stream.iter_array(X):
    ...     print(scaler.transform_one(x))
    {0: 0.8, 1: 0.2, 2: 0.4, 3: 0.4}
    {0: 0.1, 1: 0.3, 2: 0.9, 3: 0.3}
    {0: 0.5, 1: 0.7, 2: 0.5, 3: 0.1}

    """

    def __init__(self, order=2):
        self.order = order

    def transform_one(self, x):
        norm = utils.math.norm(x, order=self.order)
        return {i: xi / norm for i, xi in x.items()}


class AdaptiveStandardScaler(base.Transformer):
    """Scales data using exponentially weighted moving average and variance.

    Under the hood, a exponentially weighted running mean and variance are maintained for each
    feature. This can potentially provide better results for drifting data in comparison to
    `preprocessing.StandardScaler`. Indeed, the latter computes a global mean and variance for each
    feature, whereas this scaler weights data in proportion to their recency.

    Parameters
    ----------
    fading_factor
        This parameter is passed to `stats.EWVar`. It is expected to be in [0, 1]. More weight is
        assigned to recent samples the closer `fading_factor` is to 1.

    Examples
    --------

    Consider the following series which contains a positive trend.

    >>> import random

    >>> random.seed(42)
    >>> X = [
    ...     {'x': random.uniform(4 + i, 6 + i)}
    ...     for i in range(8)
    ... ]
    >>> for x in X:
    ...     print(x)
    {'x': 5.278}
    {'x': 5.050}
    {'x': 6.550}
    {'x': 7.446}
    {'x': 9.472}
    {'x': 10.353}
    {'x': 11.784}
    {'x': 11.173}

    This scaler works well with this kind of data because it uses statistics that assign higher
    weight to more recent data.

    >>> from river import preprocessing

    >>> scaler = preprocessing.AdaptiveStandardScaler(fading_factor=.6)

    >>> for x in X:
    ...     print(scaler.learn_one(x).transform_one(x))
    {'x': 0.0}
    {'x': -0.816}
    {'x': 0.812}
    {'x': 0.695}
    {'x': 0.754}
    {'x': 0.598}
    {'x': 0.651}
    {'x': 0.124}

    """

    def __init__(self, fading_factor=0.3):
        self.fading_factor = fading_factor
        self.vars = collections.defaultdict(functools.partial(stats.EWVar, self.fading_factor))
        self.means = collections.defaultdict(functools.partial(stats.EWMean, self.fading_factor))

    def learn_one(self, x):
        for i, xi in x.items():
            self.vars[i].update(xi)
            self.means[i].update(xi)
        return self

    def transform_one(self, x):
        return {
            i: safe_div(x[i] - m, s2**0.5 if s2 > 0 else 0)
            for i, m, s2 in ((i, self.means[i].get(), self.vars[i].get()) for i in x)
        }
