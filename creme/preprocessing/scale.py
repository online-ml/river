import collections

from .. import base
from .. import stats
from .. import utils


__all__ = ['MinMaxScaler', 'Normalizer', 'StandardScaler']


def safe_div(a, b):
    if b == 0:
        return a
    return a / b


class StandardScaler(base.Transformer):
    """Scales the data so that it has zero mean and unit variance.

    Under the hood a running mean and a running variance are maintained. The scaling is slightly
    different than when using scikit-learn but this doesn't seem to have any impact on learning
    performance.

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
              {'x': -0.27053221516496234}
              {'x': -1.3161741269826}
              {'x': -1.051218830939811}
              {'x': -1.1096762396286515}
              {'x': 1.09141268517007}
              {'x': 0.31090608502985784}
              {'x': 0.5949866915752465}
              {'x': -1.3540960661327464}
              {'x': 1.2959176347038681}
              {'x': 0.8426620966002304}
              {'x': -0.8843412116857523}
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

    def __init__(self):
        self.variances = collections.defaultdict(stats.Var)

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.variances[i].update(xi)

        return self

    def transform_one(self, x):
        return {
            i: safe_div(xi - self.variances[i].mean.get(), self.variances[i].get() ** .5)
            for i, xi in x.items()
        }


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
            i: safe_div(xi - self.min[i].get(), self.max[i].get() - self.min[i].get())
            for i, xi in x.items()
        }


class Normalizer(base.Transformer):
    """Scales a set of features so that it has unit norm.

    This is particularly useful when used after a `feature_extraction.TFIDFVectorizer`.

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
        norm = utils.norm(x, order=self.order)
        return {i: xi / norm for i, xi in x.items()}
