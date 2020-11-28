import collections
import math
import random

from river import base


class RBFSampler(base.Transformer):
    """Extracts random features which approximate an RBF kernel.

    This is a powerful way to give non-linear capacity to linear classifiers. This method is also
    called "random Fourier features" in the literature.

    Parameters
    ----------
    gamma
        RBF kernel parameter in `(-gamma * x^2)`.
    n_components
        Number of samples per original feature. Equals the dimensionality of the computed feature
        space.
    seed
        Random number seed.

    Examples
    --------

    >>> from river import feature_extraction as fx
    >>> from river import linear_model as lm
    >>> from river import optim
    >>> from river import stream

    >>> # XOR function
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> Y = [0, 0, 1, 1]

    >>> model = lm.LogisticRegression(optimizer=optim.SGD(.1))

    >>> for x, y in stream.iter_array(X, Y):
    ...     model = model.learn_one(x, y)
    ...     y_pred = model.predict_one(x)
    ...     print(y, int(y_pred))
    0 0
    0 0
    1 0
    1 1

    >>> model = (
    ...     fx.RBFSampler(seed=3) |
    ...     lm.LogisticRegression(optimizer=optim.SGD(.1))
    ... )

    >>> for x, y in stream.iter_array(X, Y):
    ...     model = model.learn_one(x, y)
    ...     y_pred = model.predict_one(x)
    ...     print(y, int(y_pred))
    0 0
    0 0
    1 1
    1 1

    References
    ----------
    [^1]: [Rahimi, A. and Recht, B., 2008. Random features for large-scale kernel machines. In Advances in neural information processing systems (pp. 1177-1184](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf)

    """

    def __init__(self, gamma=1.0, n_components=100, seed: int = None):
        self.gamma = gamma
        self.n_components = n_components
        self.seed = seed
        self.rng = random.Random(seed)
        self.weights = collections.defaultdict(self._random_weights)
        self.offsets = [random.uniform(0, 2 * math.pi) for _ in range(n_components)]

    def _random_weights(self):
        return [
            math.sqrt(2 * self.gamma) * self.rng.gauss(mu=0, sigma=1)
            for _ in range(self.n_components)
        ]

    def transform_one(self, x, y=None):
        return {
            (i, j): math.cos(xi * wj + self.offsets[j])
            for i, xi in x.items()
            for j, wj in enumerate(self.weights[i])
        }
