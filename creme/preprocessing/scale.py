import numbers
import functools
import collections

from creme import base
from creme import stats
from creme import utils


__all__ = [
    'Binarizer',
    'MaxAbsScaler',
    'MinMaxScaler',
    'Normalizer',
    'RobustScaler',
    'StandardScaler'
]


def safe_div(a, b):
    if b == 0:
        return a
    return a / b


class Binarizer(base.Transformer):
    """Binarizes the data to 0 or 1 according to a threshold.

    Parameters:
        threshold: Values above this are replaced by 1 and the others by 0.
        dtype: The desired data type to apply.

    Example:

        >>> import creme
        >>> import numpy as np

        >>> rng = np.random.RandomState(42)
        >>> X = [{'x1': v, 'x2': int(v)} for v in rng.uniform(low=-4, high=4, size=6)]

        >>> binarizer = creme.preprocessing.Binarizer()
        >>> for x in X:
        ...     print(binarizer.fit_one(x).transform_one(x))
        {'x1': False, 'x2': False}
        {'x1': True, 'x2': True}
        {'x1': True, 'x2': True}
        {'x1': True, 'x2': False}
        {'x1': False, 'x2': False}
        {'x1': False, 'x2': False}

    """

    def __init__(self, threshold=0., dtype=bool):
        self.threshold = threshold
        self.dtype = dtype

    def transform_one(self, x):
        x_tf = x.copy()

        for i, xi in x_tf.items():
            if isinstance(xi, numbers.Number):
                x_tf[i] = self.dtype(xi > self.threshold)

        return x_tf


class StandardScaler(base.Transformer):
    """Scales the data so that it has zero mean and unit variance.

    Under the hood, a running mean and a running variance are maintained. The scaling is slightly
    different than when scaling the data in batch because the exact means and variances are not
    known in advance. However, this doesn't have a detrimental impact on performance in the long
    run.

    Parameters:
        with_mean: Whether to centre the data before scaling.
        with_std: Whether to scale the data.

    Attributes:
        variances (dict): Mapping between features and instances of `stats.Var`.

    Example:

        >>> from pprint import pprint
        >>> import random
        >>> from creme import preprocessing

        >>> random.seed(42)
        >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
        >>> pprint(X)
        [{'x': 10.557707},
         {'x': 8.100043},
         {'x': 9.100117},
         {'x': 8.892842},
         {'x': 10.945884}]

        >>> scaler = preprocessing.StandardScaler()

        >>> for x in X:
        ...     print(scaler.fit_one(x).transform_one(x))
        {'x': 0.0}
        {'x': -0.707106}
        {'x': -0.123395}
        {'x': -0.263247}
        {'x': 1.195476}

    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.variances = collections.defaultdict(stats.Var)

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.variances[i].update(xi)

        return self

    def transform_one(self, x):
        x_tf = {}

        for i, xi in x.items():
            x_tf[i] = xi
            if self.with_mean:
                x_tf[i] -= self.variances[i].mean.get()
            if self.with_std:
                x_tf[i] = safe_div(x_tf[i], self.variances[i].get() ** .5)

        return x_tf


class MinMaxScaler(base.Transformer):
    """Scales the data to a fixed range from 0 to 1.

    Under the hood a running min and a running peak to peak (max - min) are maintained.

    Attributes:
        min (dict): Mapping between features and instances of `stats.Min`.
        max (dict): Mapping between features and instances of `stats.Max`.

    Example:

        >>> from pprint import pprint
        >>> import random
        >>> from creme import preprocessing

        >>> random.seed(42)
        >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
        >>> pprint(X)
        [{'x': 10.557707},
         {'x': 8.100043},
         {'x': 9.100117},
         {'x': 8.892842},
         {'x': 10.945884}]

        >>> scaler = preprocessing.MinMaxScaler()

        >>> for x in X:
        ...     print(scaler.fit_one(x).transform_one(x))
        {'x': 0.0}
        {'x': 0.0}
        {'x': 0.406920}
        {'x': 0.322582}
        {'x': 1.0}

    """

    def __init__(self):
        self.min = collections.defaultdict(stats.Min)
        self.max = collections.defaultdict(stats.Max)

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.min[i].update(xi)
            self.max[i].update(xi)

        return self

    def transform_one(self, x):
        return {
            i: safe_div(xi - self.min[i].get(),
                        self.max[i].get() - self.min[i].get())
            for i, xi in x.items()
        }


class MaxAbsScaler(base.Transformer):
    """Scales the data to a [-1, 1] range based on absolute maximum.

    Under the hood a running absolute max is maintained. This scaler is meant for
    data that is already centered at zero or sparse data. It does not shift/center
    the data, and thus does not destroy any sparsity.

    Attributes:
        abs_max (dict): Mapping between features and instances of `stats.AbsMax`.

    Example:

        >>> from pprint import pprint
        >>> import random
        >>> from creme import preprocessing

        >>> random.seed(42)
        >>> X = [{'x': random.uniform(8, 12)} for _ in range(5)]
        >>> pprint(X)
        [{'x': 10.557707},
         {'x': 8.100043},
         {'x': 9.100117},
         {'x': 8.892842},
         {'x': 10.945884}]

        >>> scaler = preprocessing.MaxAbsScaler()

        >>> for x in X:
        ...     print(scaler.fit_one(x).transform_one(x))
        {'x': 1.0}
        {'x': 0.767216}
        {'x': 0.861940}
        {'x': 0.842308}
        {'x': 1.0}

    """

    def __init__(self):
        self.abs_max = collections.defaultdict(stats.AbsMax)

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.abs_max[i].update(xi)

        return self

    def transform_one(self, x):
        return {
            i: safe_div(xi, self.abs_max[i].get())
            for i, xi in x.items()
        }


class RobustScaler(base.Transformer):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the
    interquantile range.

    Parameters:
        with_centering: Whether to centre the data before scaling.
        with_scaling: Whether to scale data to IQR.
        q_inf: Desired inferior quantile, must be between 0 and 1.
        q_sup: Desired superior quantile, must be between 0 and 1.

    Attributes:
        median (dict): Mapping between features and instances of `stats.Quantile(0.5)`.
        iqr (dict): Mapping between features and instances of `stats.IQR`.

    Example:

        >>> from pprint import pprint
        >>> import random
        >>> from creme import preprocessing

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
        ...     print(scaler.fit_one(x).transform_one(x))
        {'x': 0.0}
        {'x': -1.0}
        {'x': 0.0}
        {'x': -0.124499}
        {'x': 1.108659}

    """

    def __init__(self, with_centering=True, with_scaling=True, q_inf=.25, q_sup=.75):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.q_inf = q_inf
        self.q_sup = q_sup
        self.median = collections.defaultdict(functools.partial(stats.Quantile, 0.5))
        self.iqr = collections.defaultdict(functools.partial(stats.IQR, self.q_inf, self.q_sup))

    def fit_one(self, x, y=None):

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

    Parameters:
        order: Order of the norm (e.g. 2 corresponds to the $L^2$ norm).

    Example:

        >>> from creme import preprocessing
        >>> from creme import stream

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
