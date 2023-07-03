from __future__ import annotations

import collections
import random

from river import base


class GaussianRandomProjector(base.Transformer):
    """Gaussian random projector.

    This transformer reduces the dimensionality of inputs through Gaussian random projection.

    The components of the random projections matrix are drawn from `N(0, 1 / n_components)`.

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
    MAE: 0.933502

    References
    ----------
    [^1]: [Gaussian random projection](https://www.wikiwand.com/en/Gaussian_random_projection#Gaussian_random_projection)
    [^2]: [scikit-learn random projections module](https://scikit-learn.org/stable/modules/random_projection.html)

    """

    def __init__(self, n_components=10, seed: int | None = None):
        self.n_components = n_components
        self.seed = seed
        self._rng = random.Random(seed)
        self._projection_matrix: collections.defaultdict[
            base.typing.FeatureName, float
        ] = collections.defaultdict(self._rand_gauss)

    def _rand_gauss(self):
        return self._rng.gauss(0, 1 / (self.n_components**0.5))

    def transform_one(self, x):
        return {
            i: sum(self._projection_matrix[(i, j)] * x[j] for j in x)
            for i in range(self.n_components)
        }


class SparseRandomProjector(base.Transformer):
    """Sparse random projector.

    This transformer reduces the dimensionality of inputs by projecting them onto a sparse random
    projection matrix.

    Ping Li et al. recommend using a minimum density of `1 / sqrt(n_features)`. The transformer
    is not aware of how many features will be seen, so the user must specify the density manually.

    Parameters
    ----------
    n_components
        Number of components to project the data onto.
    density
        Density of the random projection matrix. The density is defined as the ratio of non-zero
        components in the matrix. It is equal to `1 - sparsity`.
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
    >>> model = preprocessing.SparseRandomProjector(
    ...     n_components=3,
    ...     seed=42
    ... )

    >>> for x, y in dataset:
    ...     x = model.transform_one(x)
    ...     print(x)
    ...     break
    {0: 92.89572746525327, 1: 1344540.5692342375, 2: 0}

    >>> model = (
    ...     preprocessing.SparseRandomProjector(
    ...         n_components=5,
    ...         seed=42
    ...     ) |
    ...     preprocessing.StandardScaler() |
    ...     linear_model.LinearRegression()
    ... )
    >>> evaluate.progressive_val_score(dataset, model, metrics.MAE())
    MAE: 1.292572

    References
    ----------
    [^1]: D. Achlioptas. 2003. Database-friendly random projections: Johnson-Lindenstrauss with binary coins. Journal of Computer and System Sciences 66 (2003) 671-687
    [^2]: Ping Li, Trevor J. Hastie, and Kenneth W. Church. 2006. Very sparse random projections. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD'06). ACM, New York, NY, USA, 287-296.

    """

    def __init__(self, n_components=10, density=0.1, seed: int | None = None):
        self.n_components = n_components
        self.density = density
        self.seed = seed
        self._rng = random.Random(seed)
        self._projection_matrix: collections.defaultdict[
            base.typing.FeatureName, dict[int, float]
        ] = collections.defaultdict(self._rand_weights_for_feature)

    def _rand_weights_for_feature(self):
        weights = {}
        for j in range(self.n_components):
            if self._rng.random() < self.density:
                w = (1 / (self.density * self.n_components)) ** 0.5
                # Flip a coin to decide the sign
                weights[j] = w if self._rng.random() < 0.5 else -w
        return weights

    def transform_one(self, x):
        output = {i: 0 for i in range(self.n_components)}
        for j in x:
            for i, w in self._projection_matrix[j].items():
                output[i] += w * x[j]
        return output
