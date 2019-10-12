import collections
import random

from .. import base
from .. import utils


__all__ = ['KMeans']


class KMeans(base.Clusterer):
    """Incremental k-means.

    The most common way to implement batch k-means is to use Lloyd's algorithm, which consists in
    assigning all the data points to a set of cluster centers and then moving the centers
    accordingly. This requires multiple passes over the data and thus isn't applicable in a
    streaming setting.

    In this implementation we start by finding the cluster that is closest to the current
    observation. We then move the cluster's central position towards the new observation. The
    ``halflife`` parameter determines by how much to move the cluster toward the new observation.
    You will get better results if you scale your data appropriately.

    Parameters:
        n_clusters (int): Maximum number of clusters to assign.
        halflife (float): Amount by which to move the cluster centers, a reasonable value if
            between 0 and 1.
        mu (float): Mean of the normal distribution used to instantiate cluster positions.
        sigma (float): Standard deviation of the normal distribution used to instantiate cluster
            positions.
        p (int): Power parameter for the Minkowski metric. When ``p=1``, this corresponds to the
            Manhattan distance, while ``p=2`` corresponds to the Euclidean distance.
        seed (int): Random seed used for generating initial centroid positions.

    Attributes:
        centers (dict): Central positions of each cluster.

    Example:

        In the following example the cluster assignments are exactly the same as when using
        ``sklearn``'s batch implementation. However changing the ``halflife`` parameter will
        produce different outputs.

        ::

            >>> from creme import cluster
            >>> from creme import stream

            >>> X = [
            ...     [1, 2],
            ...     [1, 4],
            ...     [1, 0],
            ...     [4, 2],
            ...     [4, 4],
            ...     [4, 0]
            ... ]

            >>> k_means = cluster.KMeans(n_clusters=2, halflife=0.4, sigma=3, seed=0)

            >>> for i, (x, _) in enumerate(stream.iter_array(X)):
            ...     k_means = k_means.fit_one(x)
            ...     print(f'{X[i]} is assigned to cluster {k_means.predict_one(x)}')
            [1, 2] is assigned to cluster 1
            [1, 4] is assigned to cluster 1
            [1, 0] is assigned to cluster 0
            [4, 2] is assigned to cluster 0
            [4, 4] is assigned to cluster 0
            [4, 0] is assigned to cluster 0

            >>> k_means.predict_one({0: 0, 1: 0})
            1

            >>> k_means.predict_one({0: 4, 1: 4})
            0

    References:
        1. `Sequential k-Means Clustering <http://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm>`_
        2. `Web-Scale K-Means Clustering <https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf>`_

    """

    def __init__(self, n_clusters=5, halflife=0.5, mu=0, sigma=1, p=2, seed=None):
        self.n_clusters = n_clusters
        self.halflife = halflife
        self.mu = mu
        self.sigma = sigma
        self.p = p
        self.rng = random.Random(seed)
        self.centers = {
            i: collections.defaultdict(self.random_normal)
            for i in range(n_clusters)
        }

    def random_normal(self):
        """Returns a random value sampled from a normal distribution."""
        return self.rng.gauss(self.mu, self.sigma)

    def fit_predict_one(self, x, y=None):
        """Equivalent to ``k_means.fit_one(x).predict_one(x)``, but faster."""

        # Find the cluster with the closest center
        closest = self.predict_one(x)

        # Move the cluster's center
        for i, xi in x.items():
            self.centers[closest][i] += self.halflife * (xi - self.centers[closest][i])

        return closest

    def fit_one(self, x, y=None):
        self.fit_predict_one(x)
        return self

    def predict_one(self, x):

        def get_distance(c):
            return utils.minkowski_distance(a=self.centers[c], b=x, p=self.p)

        return min(self.centers, key=get_distance)
