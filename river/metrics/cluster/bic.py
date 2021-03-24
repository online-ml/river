import math

from river import metrics

from . import base


class BIC(base.InternalMetric):
    r"""Bayesian Information Criterion (BIC).

    In statistics, the Bayesian Information Criterion (BIC) [^1], or Schwarz Information
    Criterion (SIC), is a criterion for model selection among a finite set of models;
    the model with the highest BIC is preferred. It is based, in part, on the likelihood
    function and is closely related to the Akaike Information Criterion (AIC).

    Let

    * k being the number of clusters,

    * $n_i$ being the number of points within each cluster, $n_1 + n_2 + ... + n_k = n$,

    * $d$ being the dimension of the clustering problem.

    Then, the variance of the clustering solution will be calculated as

    $$
    \hat{\sigma}^2 = \frac{1}{(n - m) \times d} \sum_{i = 1}^n \lVert x_i - c_j \rVert^2.
    $$

    The maximum likelihood function, used in the BIC version of `River`, would be

    $$
    \hat{l}(D) = \sum_{i = 1}^k n_i \log(n_i) - n \log n - \frac{n_i \times d}{2} \times \log(2 \pi \hat{\sigma}^2) - \frac{(n_i - 1) \times d}{2},
    $$

    and the BIC will then be calculated as

    $$
    BIC = \hat{l}(D) - 0.5 \times k \times log(n) \times (d+1).
    $$

    Using the previously mentioned maximum likelihood function, the higher the BIC value, the
    better the clustering solution is. Moreover, the BIC calculated will always be less than 0 [^2].


    Examples
    --------

    >>> from river import cluster
    >>> from river import stream
    >>> from river import metrics

    >>> X = [
    ...     [1, 2],
    ...     [1, 4],
    ...     [1, 0],
    ...     [4, 2],
    ...     [4, 4],
    ...     [4, 0],
    ...     [-2, 2],
    ...     [-2, 4],
    ...     [-2, 0]
    ... ]

    >>> k_means = cluster.KMeans(n_clusters=3, halflife=0.4, sigma=3, seed=0)
    >>> metric = metrics.cluster.BIC()

    >>> for x, _ in stream.iter_array(X):
    ...     k_means = k_means.learn_one(x)
    ...     y_pred = k_means.predict_one(x)
    ...     metric = metric.update(x, y_pred, k_means.centers)

    >>> metric
    BIC: -30.060416

    References
    ----------
    [^1]: Wikipedia contributors. (2020, December 14). Bayesian information criterion.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Bayesian_information_criterion&oldid=994127616
    [^2]: BIC Notes, https://github.com/bobhancock/goxmeans/blob/master/doc/BIC_notes.pdf

    """

    def __init__(self):
        super().__init__()
        self._ssw = metrics.cluster.SSW()
        self._n_points_by_clusters = {}
        self._n_clusters = 0
        self._dim = 0
        self._initialized = False

    def update(self, x, y_pred, centers, sample_weight=1.0):

        if not self._initialized:
            self._dim = len(x)

        self._ssw.update(x, y_pred, centers, sample_weight)

        try:
            self._n_points_by_clusters[y_pred] += 1
        except KeyError:
            self._n_points_by_clusters[y_pred] = 1

        self._n_clusters = len(centers)

        return self

    def revert(self, x, y_pred, centers, sample_weight=1.0):

        self._ssw.revert(x, y_pred, centers, sample_weight)

        self._n_points_by_clusters[y_pred] -= 1

        self._n_clusters = len(centers)

        return self

    def get(self):

        BIC = 0

        total_points = sum(self._n_points_by_clusters.values())

        try:
            variance = (
                1 / (total_points - self._n_clusters) / self._dim * self._ssw.get()
            )
        except ZeroDivisionError:
            return -math.inf

        const_term = 0.5 * self._n_clusters * math.log(total_points) * (self._dim + 1)

        for i in self._n_points_by_clusters:
            try:
                BIC += (
                    self._n_points_by_clusters[i]
                    * math.log(self._n_points_by_clusters[i])
                    - self._n_points_by_clusters[i] * math.log(total_points)
                    - (self._n_points_by_clusters[i] * self._dim)
                    / 2
                    * math.log(2 * math.pi * variance)
                    - (self._n_points_by_clusters[i] - 1) * self._dim / 2
                )
            except ValueError:
                continue

        BIC -= const_term

        return BIC

    @property
    def bigger_is_better(self):
        return True
