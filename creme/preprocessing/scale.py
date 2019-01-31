import collections

import numpy as np

from .. import base
from .. import stats


__all__ = ['StandardScaler']


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
         {'x': 0.999999...},
         {'x': 0.194726...},
         {'x': -0.312383...},
         {'x': -1.471527...},
         {'x': -1.151552...},
         {'x': -1.198587...},
         {'x': 1.166769...},
         {'x': 0.329765...},
         {'x': 0.627171...},
         {'x': -1.420187...},
         {'x': 1.353541...},
         {'x': 0.877070...},
         {'x': -0.917724...},
         {'x': -0.943887...}]

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
        self.variances = collections.defaultdict(lambda: stats.Variance())
        self.eps = np.finfo(float).eps

    def fit_one(self, x, y=None):

        for i, xi in x.items():
            self.variances[i].update(xi)

        return self.transform_one(x)

    def transform_one(self, x):
        return {
            i: (xi - self.variances[i].mean.get()) / (self.variances[i].get() + self.eps) ** 0.5
            for i, xi in x.items()
        }
