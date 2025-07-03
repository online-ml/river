from __future__ import annotations

import collections
import functools
import random

from river import base

__all__ = ["KMeans"]


class KMeans(base.Clusterer):
    """Incremental k-means.

    The most common way to implement batch k-means is to use Lloyd's algorithm, which consists in
    assigning all the data points to a set of cluster centers and then moving the centers
    accordingly. This requires multiple passes over the data and thus isn't applicable in a
    streaming setting.

    In this implementation we start by finding the cluster that is closest to the current
    observation. We then move the cluster's central position towards the new observation. The
    `halflife` parameter determines by how much to move the cluster toward the new observation.
    You will get better results if you scale your data appropriately.

    Parameters
    ----------
    n_clusters
        Maximum number of clusters to assign.
    halflife
        Amount by which to move the cluster centers, a reasonable value is between 0 and 1.
    mu
        Mean of the normal distribution used to instantiate cluster positions.
    sigma
        Standard deviation of the normal distribution used to instantiate cluster positions.
    p
        Power parameter for the Minkowski metric. When `p=1`, this corresponds to the Manhattan
        distance, while `p=2` corresponds to the Euclidean distance.
    seed
        Random seed used for generating initial centroid positions.

    Attributes
    ----------
    centers : dict
        Central positions of each cluster.

    Examples
    --------

    In the following example the cluster assignments are exactly the same as when using
    `sklearn`'s batch implementation. However, changing the `halflife` parameter will
    produce different outputs.

    >>> from river import cluster
    >>> from river import stream

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [-4, 2],
    ...     [-4, 4],
    ...     [-4, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.1, sigma=3, seed=42)

    >>> for i, (x, _) in enumerate(stream.iter_array(X)):
    ...     k_means.learn_one(x)
    ...     print(f'{X[i]} is assigned to cluster {k_means.predict_one(x)}')
    [1, 2] is assigned to cluster 1
    [1, 4] is assigned to cluster 1
    [1, 0] is assigned to cluster 0
    [-4, 2] is assigned to cluster 1
    [-4, 4] is assigned to cluster 1
    [-4, 0] is assigned to cluster 0

    >>> k_means.predict_one({0: 0, 1: 0})
    0

    >>> k_means.predict_one({0: 4, 1: 4})
    1

    References
    ----------
    [^1]: [Sequential k-Means Clustering](http://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm)
    [^2]: [Sculley, D., 2010, April. Web-scale k-means clustering. In Proceedings of the 19th international conference on World Wide Web (pp. 1177-1178)](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)

    """

    def __init__(self, n_clusters=5, halflife=0.5, mu=0, sigma=1, p=2, seed: int | None = None):
        self.n_clusters = n_clusters
        self.halflife = halflife
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.seed = seed
        self._rng = random.Random(seed)
        rand_gauss = functools.partial(self._rng.gauss, self.mu, self.sigma)
        self.centers: dict[int, collections.defaultdict] = {
            i: collections.defaultdict(rand_gauss) for i in range(n_clusters)
        }

    @property
    def _mutable_attributes(self):
        return {"halflife"}

    def learn_predict_one(self, x):
        """Equivalent to `k_means.learn_one(x).predict_one(x)`, but faster."""

        # Find the cluster with the closest center
        closest = self.predict_one(x)

        # Move the cluster's center
        for i, xi in x.items():
            self.centers[closest][i] += self.halflife * (xi - self.centers[closest][i])

        return closest

    def learn_one(self, x):
        self.learn_predict_one(x)

    def predict_one(self, x):
        def get_distance(c):
            center = self.centers[c]
            return sum(
                (abs(center[k] - x.get(k, 0))) ** self.p for k in {*center.keys(), *x.keys()}
            )

        return min(self.centers, key=get_distance)

    @classmethod
    def _unit_test_params(cls):
        yield {"n_clusters": 5}
