import collections

import numpy as np

from .. import base
from .. import stats


__all__ = ['MinMaxScaler', 'StandardScaler']


class StandardScaler(base.Transformer):
    """Scales the data so that it has mean 0 and variance 1.

    Under the hood a running mean and a running variance are maintained. The scaling is slightly
    different than when using scikit-learn but this doesn't seem to have any impact learning
    performance.

    Attributes:
        variances (dict): Mapping between features and instances of ``stats.Variance``.
        eps (float): Used for avoiding divisions by zero.

    Example:

    ::

        >>> import pprint
        >>> import creme
        >>> import numpy as np
        >>> from sklearn import preprocessing

        >>> rng = np.random.RandomState(42)
        >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

        >>> scaler = creme.preprocessing.StandardScaler()
        >>> pprint.pprint([scaler.fit_one(x) for x in X])
        [{'x': 0.0},
         {'x': 0.7071067811865472},
         {'x': 0.15899361505958232},
         {'x': -0.2705322151649623},
         {'x': -1.3161741269825997},
         {'x': -1.0512188309398107},
         {'x': -1.1096762396286515},
         {'x': 1.09141268517007},
         {'x': 0.3109060850298578},
         {'x': 0.5949866915752465},
         {'x': -1.3540960661327461},
         {'x': 1.2959176347038681},
         {'x': 0.8426620966002304},
         {'x': -0.8843412116857523},
         {'x': -0.91188180555936}]


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
        self.variances = collections.defaultdict(stats.Variance)
        self.eps = np.finfo(float).eps

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.variances[i].update(xi)

        return self.transform_one(x)

    def transform_one(self, x):
        return {
            i: (xi - (self.variances[i].mean.get() or 0)) / (self.variances[i].get() + self.eps) ** 0.5
            for i, xi in x.items()
        }


class MinMaxScaler(base.Transformer):
    """Scales the data to a fixed range 0 to 1.

    Under the hood a running min and a running peak to peak (max - min) are maintained. The scaling is slightly
    different than when using scikit-learn but this doesn't seem to have any impact learning
    performance.

    Attributes:
        min (dict): Mapping between features and instances of ``stats.Min``.
        max (dict): Mapping between features and instances of ``stats.Max``.
        eps (float): Used for avoiding divisions by zero.

    Example:

        >>> import pprint
        >>> import creme
        >>> import numpy as np
        >>> from sklearn import preprocessing

        >>> rng = np.random.RandomState(42)
        >>> X = [{'x': v} for v in rng.uniform(low=8, high=12, size=15)]

        >>> scaler = creme.preprocessing.MinMaxScaler()
        >>> pprint.pprint([scaler.fit_one(x) for x in X])
        [{'x': 0.0},
         {'x': 1.0},
         {'x': 0.620391...},
         {'x': 0.388976...},
         {'x': 0.0},
         {'x': 0.0},
         {'x': 0.0},
         {'x': 0.905293...},
         {'x': 0.608349...},
         {'x': 0.728172...},
         {'x': 0.0},
         {'x': 1.0},
         {'x': 0.855194...},
         {'x': 0.201990...},
         {'x': 0.169847...}]

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
        self.eps = np.finfo(float).eps

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.min[i].update(xi)
            self.max[i].update(xi)

        return self.transform_one(x)

    def transform_one(self, x):
        return {
            i: (xi - self.min[i].get()) / (self.max[i].get() - self.min[i].get() + self.eps)
            for i, xi in x.items()
        }
