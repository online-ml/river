import math

from scipy.special import factorial

from . import base

__all__ = ["Q0", "Q2"]


class Q0(base.MultiClassMetric):
    r"""Q0 index.

    Dom's Q0 measure [^2] uses conditional entropy to calculate the goodness of
    a clustering solution. However, this term only evaluates the homogeneity of
    a solution. To measure the completeness of the hypothesized clustering, Dom
    includes a model cost term calculated using a coding theory argument. The
    overall clustering quality measure presented is the sum of the costs of
    representing the data's conditional entropy and the model.

    The motivation for this approach is an appeal to parsimony: Given identical
    conditional entropies, H(C|K), the clustering solution with the fewest clusters
    should be preferred.

    The Q0 measure can be calculated using the following formula [^1]

    $$
    Q_0(C, K) = H(C|K) + \frac{1}{n} \sum_{k=1}^{|K|} \log \binom{h(c) + |C| - 1}{|C| - 1}.
    $$

    Due to the complexity of the formula, this metric and its associated normalized version (Q2)
    is one order of magnitude slower than most other implemented metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.Q0()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.9182958340544896
    1.208582260960826
    1.4479588303902937
    1.3803939544277863

    >>> metric
    Q0: 1.380394

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007. URL: https://www.aclweb.org/anthology/D07-1043.pdf.
    [^2]: Byron E. Dom. 2001. An information-theoretic external
          cluster-validity measure. Technical Report RJ10219, IBM, October.

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    @staticmethod
    def binomial_coeff(n, k):
        return factorial(n) / (factorial(k) * factorial(n - k))

    def get(self):

        conditional_entropy_c_k = 0.0

        sum_logs = 0.0

        n_true_clusters = sum(1 for i in self.cm.sum_col.values() if i > 0)

        for i in self.cm.classes:

            for j in self.cm.classes:

                try:
                    conditional_entropy_c_k -= (
                        self.cm[j][i]
                        / self.cm.n_samples
                        * math.log(self.cm[j][i] / self.cm.sum_col[i], 2)
                    )
                except (ValueError, ZeroDivisionError):
                    continue

            try:
                sum_logs += math.log(
                    self.binomial_coeff(
                        self.cm.sum_col[i] + n_true_clusters - 1, n_true_clusters - 1
                    )
                )
            except ValueError:
                continue

        return conditional_entropy_c_k + sum_logs / self.cm.n_samples


class Q2(Q0):
    r"""Q2 index.

    Q2 index is presented by Dom [^2] as a normalized version of the original Q0 index.
    This index has a range of $(0, 1]$ [^1], with greater scores being representing more
    preferred clustering.

    The Q2 index can be calculated as follows [^1]

    $$
    Q2(C, K) = \frac{\frac{1}{n} \sum_{c=1}^{|C|} \log \binom{h(c) + |C| - 1}{|C| - 1} }{Q_0(C, K)}
    $$

    where $C$ is the target partition, $K$ is the hypothesized partition and $h(k)$ is
    the size of cluster $k$.

    Due to the complexity of the formula, this metric is one order of magnitude slower than
    its original version (Q0) and most other implemented metrics.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.Q2()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.0
    0.4545045563529578
    0.39923396953448914
    0.3979343306829813

    >>> metric
    Q2: 0.397934

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007. URL: https://www.aclweb.org/anthology/D07-1043.pdf.
    [^2]: Byron E. Dom. 2001. An information-theoretic external
          cluster-validity measure. Technical Report RJ10219, IBM, October.

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    def get(self):

        q0 = super().get()

        sum_logs = 0.0

        n_true_clusters = sum(1 for i in self.cm.sum_col.values() if i > 0)

        for i in self.cm.classes:

            try:
                sum_logs += math.log(
                    self.binomial_coeff(
                        self.cm.sum_row[i] + n_true_clusters - 1, n_true_clusters - 1
                    )
                )
            except ValueError:
                continue

        try:
            return (sum_logs / self.cm.n_samples) / q0
        except ZeroDivisionError:
            return 0.0
