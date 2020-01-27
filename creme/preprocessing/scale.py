import functools
import collections

from .. import base
from .. import stats
from .. import utils


__all__ = [
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


class StandardScaler(base.Transformer):
    """Scales the data so that it has zero mean and unit variance.

    Under the hood a running mean and a running variance are maintained. The scaling is slightly
    different than when using scikit-learn but this doesn't seem to have any impact on learning
    performance.

    Parameters:
        with_mean (bool): Whether to centre the data before scaling. Defaults to ``True``.
        with_std (bool): Whether to scale data. Defaults to ``True``.

    Attributes:
        variances (dict): Mapping between features and instances of `stats.Var`.

    Example:

        ::

              >>> import creme
              >>> import numpy as np
              >>> from sklearn import preprocessing

              >>> rng = np.random.RandomState(42)
              >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

              >>> scaler = creme.preprocessing.StandardScaler()
              >>> for x in X:
              ...     print(scaler.fit_one(x).transform_one(x))
              {'x': 0.0}
              {'x': 0.7071067811865474}
              {'x': 0.15899361505958234}
              {'x': -0.2705322151649623}
              {'x': -1.3161741269826}
              {'x': -1.051218830939811}
              {'x': -1.1096762396286515}
              {'x': 1.09141268517007}
              {'x': 0.3109060850298578}
              {'x': 0.5949866915752465}
              {'x': -1.3540960661327461}
              {'x': 1.2959176347038681}
              {'x': 0.8426620966002304}
              {'x': -0.8843412116857524}
              {'x': -0.91188180555936}

              >>> X = np.array([x['x'] for x in X]).reshape(-1, 1)
              >>> preprocessing.StandardScaler().fit_transform(X)
              array([[-0.36224883],
                     [ 1.37671717],
                     [ 0.71659166],
                     [ 0.31416852],
                     [-1.02177407],
                     [-1.02184687],
                     [-1.31735428],
                     [ 1.1215704 ],
                     [ 0.32158263],
                     [ 0.64439399],
                     [-1.43053132],
                     [ 1.43465174],
                     [ 1.01975844],
                     [-0.85179183],
                     [-0.94388734]])

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

    Under the hood a running min and a running peak to peak (max - min) are maintained. The scaling
    is slightly different than when using scikit-learn but this doesn't seem to have any impact on
    learning performance.

    Attributes:
        min (dict): Mapping between features and instances of `stats.Min`.
        max (dict): Mapping between features and instances of `stats.Max`.

    Example:

        ::

            >>> import creme
            >>> import numpy as np
            >>> from sklearn import preprocessing

            >>> rng = np.random.RandomState(42)
            >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

            >>> scaler = creme.preprocessing.MinMaxScaler()
            >>> for x in X:
            ...     print(scaler.fit_one(x).transform_one(x))
            {'x': 0.0}
            {'x': 1.0}
            {'x': 0.6203919416734277}
            {'x': 0.3889767542308411}
            {'x': 0.0}
            {'x': 0.0}
            {'x': 0.0}
            {'x': 0.9052932403284701}
            {'x': 0.6083494586037179}
            {'x': 0.7281723223510779}
            {'x': 0.0}
            {'x': 1.0}
            {'x': 0.8551948389216542}
            {'x': 0.20199040802352813}
            {'x': 0.16984743067826635}

            >>> X = np.array([x['x'] for x in X]).reshape(-1, 1)
            >>> preprocessing.MinMaxScaler().fit_transform(X)
            array([[0.37284965],
                   [0.9797798 ],
                   [0.74938422],
                   [0.60893137],
                   [0.14266357],
                   [0.14263816],
                   [0.03950081],
                   [0.89072903],
                   [0.61151903],
                   [0.72418595],
                   [0.        ],
                   [1.        ],
                   [0.85519484],
                   [0.20199041],
                   [0.16984743]])

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

        ::

            >>> import creme
            >>> import numpy as np
            >>> from sklearn import preprocessing

            >>> rng = np.random.RandomState(42)
            >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

            >>> scaler = creme.preprocessing.MaxAbsScaler()
            >>> for x in X:
            ...     print(scaler.fit_one(x).transform_one(x))
            {'x': 1.0}
            {'x': 1.0}
            {'x': 0.9258754518784218}
            {'x': 0.8806879332749703}
            {'x': 0.7306768519605097}
            {'x': 0.7306686776326253}
            {'x': 0.6974865739110592}
            {'x': 0.9713499336579836}
            {'x': 0.8815204528926222}
            {'x': 0.9177684779286244}
            {'x': 0.6847780857355226}
            {'x': 1.0}
            {'x': 0.9537133387191868}
            {'x': 0.7449179338112799}
            {'x': 0.734643499572489}

            >>> X = np.array([x['x'] for x in X]).reshape(-1, 1)
            >>> preprocessing.MaxAbsScaler().fit_transform(X)
            array([[0.79953273],
                [0.99353666],
                [0.9198912 ],
                [0.87499575],
                [0.72595424],
                [0.72594612],
                [0.69297848],
                [0.96507177],
                [0.87582288],
                [0.91183663],
                [0.68035213],
                [1.        ],
                [0.95371334],
                [0.74491793],
                [0.7346435 ]])
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
        with_centering (bool): Whether to centre the data before scaling. Defaults to ``True``.
        with_scaling (bool): Whether to scale data to iqr. Defaults to ``True``.
        q_inf (float): Desired inferior quantile, must be between 0 and 1. Defaults to ``0.25``.
        q_sup (float): Desired superior quantile, must be between 0 and 1. Defaults to ``0.75``.

    Attributes:
        median (dict): Mapping between features and instances of `stats.Quantile(0.5)`.
        iqr (dict): Mapping between features and instances of `stats.IQR`.

    Example:

        ::

            >>> import creme
            >>> import numpy as np
            >>> from sklearn import preprocessing

            >>> rng = np.random.RandomState(42)
            >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

            >>> scaler = creme.preprocessing.RobustScaler()
            >>> for x in X:
            ...     print(scaler.fit_one(x).transform_one(x))
            {'x': 0.0}
            {'x': 0.0}
            {'x': 0.0}
            {'x': -0.3787338518541633}
            {'x': -1.2383133577483856}
            {'x': -2.6296427694340427}
            {'x': -1.1145387387139178}
            {'x': 1.3429672438986109}
            {'x': 0.037059676555904184}
            {'x': 0.25865683357370733}
            {'x': -1.165696864901573}
            {'x': 0.7074284260047643}
            {'x': 0.25014164136194217}
            {'x': -0.8213524413984991}
            {'x': -0.6509250023240815}

            >>> X = np.array([x['x'] for x in X]).reshape(-1, 1)
            >>> preprocessing.RobustScaler().fit_transform(X)
            array([[-0.36543233],
                [ 0.57403854],
                [ 0.21740783],
                [ 0.        ],
                [-0.72173876],
                [-0.72177808],
                [-0.88142503],
                [ 0.4361963 ],
                [ 0.00400545],
                [ 0.17840326],
                [-0.94256856],
                [ 0.60533751],
                [ 0.38119272],
                [-0.62990639],
                [-0.6796607 ]])
    """

    def __init__(self, with_centering=True, with_scaling=True, q_inf=0.25, q_sup=0.75):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.median = collections.defaultdict(
            functools.partial(stats.Quantile, 0.5)
        )
        self.iqr = collections.defaultdict(
            functools.partial(stats.IQR, q_inf, q_sup)
        )

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
        order (int): Order of the norm (e.g. 2 corresponds to the $L^2$ norm).

    Example:

        ::

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
