from __future__ import annotations

import collections
import functools
import itertools
import numbers
import typing

import narwhals.stable.v2 as nw
import numpy as np

from river import base, stats, utils

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.stable.v2.typing import IntoDataFrame, IntoDataFrameT

__all__ = [
    "AdaptiveStandardScaler",
    "Binarizer",
    "MaxAbsScaler",
    "MinMaxScaler",
    "Normalizer",
    "RobustScaler",
    "StandardScaler",
]

NW_TO_NP_DTYPES: Mapping[nw.dtypes.DType, np.number] = {
    nw.Int8(): np.float16(),
    nw.Int16(): np.float32(),
    nw.Int32(): np.float64(),
    nw.UInt8(): np.float16(),
    nw.UInt16(): np.float32(),
    nw.UInt32(): np.float64(),
    nw.Float32(): np.float32(),
    nw.Float64(): np.float64(),
}


def safe_div(a, b):
    """Return a if b is nil, else divides a by b.

    When scaling, sometimes a denominator might be nil. For instance, during standard scaling
    the denominator can be nil if a feature has no variance.

    """
    return a / b if b else 0.0


class Binarizer(base.Transformer):
    """Binarize the data to 0 or 1 according to a threshold.

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
    ...     binarizer.learn_one(x)
    ...     print(binarizer.transform_one(x))
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

    When ``window_size`` is set, the running mean and variance are replaced by rolling versions
    computed over the last ``window_size`` observations via `utils.Rolling` wrapping `stats.Mean`
    and `stats.Var`. In this mode, `learn_many` is processed row by row because the mini-batch
    merge formula for variance does not yield a correct rolling estimate.

    Parameters
    ----------
    with_std
        Whether or not each feature should be divided by its standard deviation.
    window_size
        Size of the rolling window used to compute the mean and variance. If ``None``, the
        running mean and variance over the entire stream are used.

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
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
    {'x': 0.0, 'y': 0.0}
    {'x': -0.999, 'y': 0.999}
    {'x': 0.937, 'y': 1.350}
    {'x': 1.129, 'y': -0.651}
    {'x': -0.776, 'y': -0.729}
    {'x': -1.274, 'y': 0.992}

    A rolling window can be used to scale relative to the most recent observations only.
    The variance is the population variance (``ddof=0``), matching the running estimator
    used in the default mode:

    >>> scaler = preprocessing.StandardScaler(window_size=3)
    >>> for x in X:
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
    {'x': 0.0, 'y': 0.0}
    {'x': -1.0, 'y': 1.0}
    {'x': 0.937, 'y': 1.351}
    {'x': 0.983, 'y': -0.960}
    {'x': -1.337, 'y': -0.803}
    {'x': -1.036, 'y': 1.406}

    This transformer also supports mini-batch updates. You can call `learn_many` and provide a
    `pandas.DataFrame`:

    >>> import pandas as pd
    >>> X = pd.DataFrame.from_dict(X)

    >>> scaler = preprocessing.StandardScaler()
    >>> scaler.learn_many(X[:3])
    >>> scaler.learn_many(X[3:])

    You can then call `transform_many` to scale a mini-batch of features:

    >>> scaler.transform_many(X)
        x         y
    0  0.444600 -0.933384
    1 -1.044259 -0.138809
    2  0.841106  1.679208
    3  1.477301 -0.685117
    4 -0.444084 -0.914195
    5 -1.274664  0.992296

    A scaler can also be warm-started from previously computed statistics, e.g. to
    resume from a checkpoint or to seed the stream with an offline estimate:

    >>> scaler = preprocessing.StandardScaler._from_state(
    ...     counts={'x': 100},
    ...     means={'x': 10.0},
    ...     vars={'x': 4.0},
    ... )
    >>> scaler.transform_one({'x': 12.0})
    {'x': 1.0}

    References
    ----------
    [^1]: [Welford's Method (and Friends)](https://www.embeddedrelated.com/showarticle/785.php)
    [^2]: [Batch updates for simple statistics](https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html)

    """

    def __init__(self, with_std=True, window_size: int | None = None) -> None:
        self.with_std = with_std
        self.window_size = window_size
        self.counts: collections.Counter = collections.Counter()
        if window_size is None:
            self.means: collections.defaultdict = collections.defaultdict(float)
            self.vars: collections.defaultdict = collections.defaultdict(float)
        else:
            self.means = collections.defaultdict(
                functools.partial(utils.Rolling, stats.Mean, window_size=window_size)
            )
            # Use ddof=0 (population variance) to match the Welford estimator used in the
            # non-windowed branch; otherwise the two modes would disagree.
            self.vars = collections.defaultdict(
                functools.partial(utils.Rolling, stats.Var, window_size=window_size, ddof=0)
            )

    def __setstate__(self, state: dict) -> None:
        # Default `window_size` to None so pickles written before this attribute was
        # introduced keep working without re-running __init__.
        state.setdefault("window_size", None)
        self.__dict__.update(state)

    @classmethod
    def _from_state(
        cls,
        counts: dict,
        means: dict,
        vars: dict | None = None,
        *,
        with_std: bool = True,
    ) -> StandardScaler:
        """Create a new instance with pre-populated running statistics.

        Useful to warm-start a scaler from offline-computed statistics or to resume
        from a checkpoint without replaying past observations.

        Note that warm-starting a windowed scaler from scalar statistics is not
        supported because a single mean/variance cannot reconstruct the underlying
        window of observations; replay the recent observations through ``learn_one``
        instead.

        Parameters
        ----------
        counts
            Mapping between features and the number of observations they have been
            updated with.
        means
            Mapping between features and their running mean.
        vars
            Mapping between features and their running variance. Required when
            ``with_std`` is ``True``; ignored otherwise.
        with_std
            Whether or not each feature should be divided by its standard deviation.

        """
        new = cls(with_std=with_std)
        new.counts.update(counts)
        new.means.update(means)
        if with_std and vars is not None:
            new.vars.update(vars)
        return new

    def learn_one(self, x):
        counts = self.counts
        means = self.means
        vars_ = self.vars
        if self.window_size is not None:
            # Rolling Mean/Var: delegate eviction logic to the underlying stats objects.
            if self.with_std:
                for i, xi in x.items():
                    counts[i] += 1
                    means[i].update(xi)
                    vars_[i].update(xi)
            else:
                for i, xi in x.items():
                    counts[i] += 1
                    means[i].update(xi)
            return
        if self.with_std:
            for i, xi in x.items():
                counts[i] += 1
                old_mean = means[i]
                means[i] += (xi - old_mean) / counts[i]
                vars_[i] += ((xi - old_mean) * (xi - means[i]) - vars_[i]) / counts[i]
        else:
            for i, xi in x.items():
                counts[i] += 1
                old_mean = means[i]
                means[i] += (xi - old_mean) / counts[i]

    def transform_one(self, x):
        means = self.means
        if self.window_size is not None:
            if self.with_std:
                vars_ = self.vars
                result = {}
                for i, xi in x.items():
                    m = means[i].get()
                    v = vars_[i].get()
                    result[i] = (xi - m) / v**0.5 if v else 0.0
                return result
            return {i: xi - means[i].get() for i, xi in x.items()}
        if self.with_std:
            vars_ = self.vars
            result = {}
            for i, xi in x.items():
                v = vars_[i]
                result[i] = (xi - means[i]) / v**0.5 if v else 0.0
            return result
        return {i: xi - means[i] for i, xi in x.items()}

    def learn_many(self, X: IntoDataFrame) -> None:
        """Update with a mini-batch of features.

        Note that the update formulas for mean and variance are slightly different than in the
        single instance case, but they produce exactly the same result. When ``window_size``
        is set, the rows are processed sequentially because the batched merge formula is not
        compatible with rolling-window eviction.

        Parameters
        ----------
        X
            A dataframe where each column is a feature. Any narwhals-supported eager backend
            (pandas, polars, pyarrow, ...) is accepted.
        """
        Xnw = utils.dataframe.into_frame(X)

        if self.window_size is not None:
            # Row-by-row to preserve correct rolling-window semantics.
            for row in Xnw.iter_rows(named=True):
                self.learn_one(row)
            return

        # Drop to a float64 numpy matrix for the compute core; the column labels drive the
        # per-feature statistics, so the batch may add/drop/reorder columns between calls.
        columns = Xnw.columns
        X_np = utils.dataframe.to_numpy(Xnw)

        # In the rest of this method, old_* refers to the existing statistics, whilst new_* refers
        # to the statistics of the current mini-batch.

        new_means = np.nanmean(X_np, axis=0)
        # We could call np.var, but we already have the mean so we can be smart
        if self.with_std:
            new_vars = np.einsum("ij,ij->j", X_np, X_np) / len(X_np) - new_means**2
        else:
            new_vars = []
        new_counts = np.sum(~np.isnan(X_np), axis=0)

        for col, new_mean, new_var, new_count in itertools.zip_longest(
            columns, new_means, new_vars, new_counts
        ):
            old_mean = self.means[col]
            old_var = self.vars[col]
            old_count = self.counts[col]

            a = old_count / (old_count + new_count)
            b = new_count / (old_count + new_count)

            self.means[col] = (a * old_mean + b * new_mean).item()
            if self.with_std:
                self.vars[col] = (
                    a * old_var + b * new_var + a * b * (old_mean - new_mean) ** 2
                ).item()
            self.counts[col] += new_count.item()

    def transform_many(self, X: IntoDataFrameT) -> IntoDataFrameT:
        """Scale a mini-batch of features.

        Every narwhals-supported backend (pandas, polars, pyarrow, nullable/arrow-backed
        pandas, ...) takes the same backend-agnostic path. The compute dtype is inferred from
        the input schema, so a feature's float dtype is preserved (e.g. ``float32`` stays
        ``float32``); integer columns are widened to the matching float and anything else falls
        back to ``float64``.

        Parameters
        ----------
        X
            A dataframe where each column is a feature. An exception will be raised if any of
            the features has not been seen during a previous call to `learn_many`.

        """
        Xnw = utils.dataframe.into_frame(X)
        schema = Xnw.schema
        columns = schema.names()
        dtypes = {NW_TO_NP_DTYPES.get(dtype, np.float64()) for dtype in schema.dtypes()}
        dtype = np.result_type(*dtypes)

        if self.window_size is None:
            means = np.array([self.means[c] for c in columns], dtype=dtype)
        else:
            means = np.array([self.means[c].get() for c in columns], dtype=dtype)

        Xt = utils.dataframe.to_numpy(Xnw, dtype=dtype) - means

        if self.with_std:
            if self.window_size is None:
                stds = np.array([self.vars[c] ** 0.5 for c in columns], dtype=dtype)
            else:
                stds = np.array([self.vars[c].get() ** 0.5 for c in columns], dtype=dtype)
            np.divide(Xt, stds, where=stds > 0, out=Xt)

        native = utils.dataframe.to_native_frame(Xt, columns=columns, like=Xnw)
        return typing.cast("IntoDataFrameT", native)


class MinMaxScaler(base.Transformer):
    """Scales the data to a fixed range from 0 to 1.

    Under the hood a running min and a running peak to peak (max - min) are maintained.
    When ``window_size`` is set, the scaler tracks the min and max over the last
    ``window_size`` observations via `stats.RollingMin` and `stats.RollingMax` instead.

    Parameters
    ----------
    window_size
        Size of the rolling window used to compute the min and max. If ``None``, the
        running min and max over the entire stream are used.

    Attributes
    ----------
    min : dict
        Mapping between features and instances of `stats.Min` (or `stats.RollingMin`
        when ``window_size`` is set).
    max : dict
        Mapping between features and instances of `stats.Max` (or `stats.RollingMax`
        when ``window_size`` is set).

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
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
    {'x': 0.0}
    {'x': 0.0}
    {'x': 0.406920}
    {'x': 0.322582}
    {'x': 1.0}

    A rolling window can be used to scale relative to the most recent observations
    only:

    >>> scaler = preprocessing.MinMaxScaler(window_size=3)
    >>> for x in X:
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
    {'x': 0.0}
    {'x': 0.0}
    {'x': 0.406920}
    {'x': 0.792741}
    {'x': 1.0}

    A scaler can also be warm-started from previously computed statistics, e.g. to
    resume from a checkpoint or to seed the stream with an offline estimate:

    >>> scaler = preprocessing.MinMaxScaler._from_state(min={'x': 8.0}, max={'x': 12.0})
    >>> scaler.transform_one({'x': 10.0})
    {'x': 0.5}

    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        if window_size is None:
            self.min: collections.defaultdict = collections.defaultdict(stats.Min)
            self.max: collections.defaultdict = collections.defaultdict(stats.Max)
        else:
            self.min = collections.defaultdict(functools.partial(stats.RollingMin, window_size))
            self.max = collections.defaultdict(functools.partial(stats.RollingMax, window_size))

    def __setstate__(self, state: dict) -> None:
        # Default `window_size` to None so pickles written before this attribute was
        # introduced keep working without re-running __init__.
        state.setdefault("window_size", None)
        self.__dict__.update(state)

    @classmethod
    def _from_state(
        cls,
        min: dict,
        max: dict,
        window_size: int | None = None,
    ) -> MinMaxScaler:
        """Create a new instance with pre-populated running min and max.

        Useful to warm-start a scaler from offline-computed statistics or to resume
        from a checkpoint without replaying past observations.

        Parameters
        ----------
        min
            Mapping between features and their initial min.
        max
            Mapping between features and their initial max.
        window_size
            Size of the rolling window, forwarded to ``__init__``. When set, each
            initial value seeds one slot of the rolling window and will eventually be
            evicted as fresh observations arrive.

        """
        new = cls(window_size=window_size)
        for k, v in min.items():
            new.min[k].update(v)
        for k, v in max.items():
            new.max[k].update(v)
        return new

    def learn_one(self, x):
        min_ = self.min
        max_ = self.max
        for i, xi in x.items():
            min_[i].update(xi)
            max_[i].update(xi)

    def transform_one(self, x):
        min_ = self.min
        max_ = self.max
        result = {}
        for i, xi in x.items():
            lo = min_[i].get()
            hi = max_[i].get()
            d = hi - lo
            result[i] = (xi - lo) / d if d else 0.0
        return result


class MaxAbsScaler(base.Transformer):
    """Scales the data to a [-1, 1] range based on absolute maximum.

    Under the hood a running absolute max is maintained. This scaler is meant for
    data that is already centered at zero or sparse data. It does not shift/center
    the data, and thus does not destroy any sparsity.

    When ``window_size`` is set, the scaler tracks the absolute max over the last
    ``window_size`` observations via `stats.RollingAbsMax` instead of the entire
    stream.

    Parameters
    ----------
    window_size
        Size of the rolling window used to compute the absolute max. If ``None``,
        the running absolute max over the entire stream is used.

    Attributes
    ----------
    abs_max : dict
        Mapping between features and instances of `stats.AbsMax` (or
        `stats.RollingAbsMax` when ``window_size`` is set).

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
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
    {'x': 1.0}
    {'x': 0.767216}
    {'x': 0.861940}
    {'x': 0.842308}
    {'x': 1.0}

    A scaler can also be warm-started from a previously computed absolute max:

    >>> scaler = preprocessing.MaxAbsScaler._from_state(abs_max={'x': 12.0})
    >>> scaler.transform_one({'x': 6.0})
    {'x': 0.5}

    """

    def __init__(self, window_size: int | None = None) -> None:
        self.window_size = window_size
        if window_size is None:
            self.abs_max: collections.defaultdict = collections.defaultdict(stats.AbsMax)
        else:
            self.abs_max = collections.defaultdict(
                functools.partial(stats.RollingAbsMax, window_size)
            )

    def __setstate__(self, state: dict) -> None:
        # Default `window_size` to None so pickles written before this attribute was
        # introduced keep working without re-running __init__.
        state.setdefault("window_size", None)
        self.__dict__.update(state)

    @classmethod
    def _from_state(
        cls,
        abs_max: dict,
        window_size: int | None = None,
    ) -> MaxAbsScaler:
        """Create a new instance with a pre-populated running absolute max.

        Useful to warm-start a scaler from an offline-computed statistic or to resume
        from a checkpoint without replaying past observations.

        Parameters
        ----------
        abs_max
            Mapping between features and their initial absolute max.
        window_size
            Size of the rolling window, forwarded to ``__init__``. When set, each
            initial value seeds one slot of the rolling window and will eventually
            be evicted as fresh observations arrive.

        """
        new = cls(window_size=window_size)
        for k, v in abs_max.items():
            new.abs_max[k].update(v)
        return new

    def learn_one(self, x):
        abs_max = self.abs_max
        for i, xi in x.items():
            abs_max[i].update(xi)

    def transform_one(self, x):
        abs_max = self.abs_max
        result = {}
        for i, xi in x.items():
            m = abs_max[i].get()
            result[i] = xi / m if m else 0.0
        return result


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
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
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

    def transform_one(self, x):
        x_tf = {}

        for i, xi in x.items():
            x_tf[i] = xi
            if self.with_centering:
                median = self.median[i].get()
                if median is not None:
                    x_tf[i] -= median
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
    ...     scaler.learn_one(x)
    ...     print(scaler.transform_one(x))
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

    def transform_one(self, x):
        return {
            i: safe_div(x[i] - m, s2**0.5 if s2 > 0 else 0)
            for i, m, s2 in ((i, self.means[i].get(), self.vars[i].get()) for i in x)
        }
