import collections
import random
from river import base


class GaussianRandomProjector(base.Transformer):
    """Dimensionality reduction through Gaussian random projection.

    Parameters
    ----------
    n_components
        Number of components to project the data onto.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.TrumpApproval()
    >>> model = preprocessing.GaussianRandomProjector(
    ...     n_components=3,
    ...     seed=42
    ... )

    >>> for x, y in dataset:
    ...     x = model.transform_one(x)
    ...     print(x)
    ...     break
    {0: -61289.37139206629, 1: 141312.51039283074, 2: 279165.99370457436}

    >>> model = (
    ...     preprocessing.GaussianRandomProjector(
    ...         n_components=5,
    ...         seed=42
    ...     ) |
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LinearRegression()
    ... )
    >>> evaluate.progressive_val_score(dataset, model, metrics.MAE())
    MAE: 0.860464

    """

    def __init__(self, n_components=10, seed: int = None):
        self.n_components = n_components
        self.seed = seed
        self._rng = random.Random(seed)
        self._projection_matrix = collections.defaultdict(self._rand_gauss)

    def _rand_gauss(self):
        return self._rng.gauss(0, 1 / (self.n_components**0.5))

    def transform_one(self, x):
        return {
            i: sum(self._projection_matrix[(i, j)] * x[j] for j in x)
            for i in range(self.n_components)
        }
