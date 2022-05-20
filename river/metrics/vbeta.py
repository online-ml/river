import math

from river import metrics

__all__ = ["Completeness", "Homogeneity", "VBeta"]


class Homogeneity(metrics.base.MultiClassMetric):
    r"""Homogeneity Score.

    Homogeneity metric [^1] of a cluster labeling given a ground truth.

    In order to satisfy the homogeneity criteria, a clustering must assign only
    those data points that are members of a single class to a single cluster. That
    is, the class distribution within each cluster should be skewed to a single
    class, that is, zero entropy. We determine how close a given clustering is to
    this ideal by examining the conditional entropy of the class distribution given
    the proposed clustering.

    However, in an imperfect situation, the size of this value is dependent on the
    size of the dataset and the distribution of class sizes. Therefore, instead of
    taking the raw conditional entropy, we normalize by the maximum reduction in
    entropy the clustering information could provide.

    As such, we define homogeneity as:

    $$
    h = \begin{cases}
    1 if H(C) = 0, \\
    1 - \frac{H(C|K)}{H(C)} otherwise.
    \end{cases}.
    $$

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

    >>> metric = metrics.Homogeneity()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    0.0
    0.311278
    0.37515
    0.42062

    >>> metric
    Homogeneity: 42.06%

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    @property
    def works_with_weights(self):
        return False

    def get(self):
        raw_conditional_entropy = 0.0
        max_reduction_entropy = 0.0

        for i in self.cm.classes:

            for j in self.cm.classes:
                try:
                    raw_conditional_entropy -= (
                        self.cm[j][i]
                        / self.cm.n_samples
                        * math.log(self.cm[j][i] / self.cm.sum_col[i], 2)
                    )
                except (ValueError, ZeroDivisionError):
                    continue

            try:
                max_reduction_entropy -= (
                    self.cm.sum_row[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_row[i] / self.cm.n_samples, 2)
                )
            except (ValueError, ZeroDivisionError):
                continue

        try:
            return 1.0 - raw_conditional_entropy / max_reduction_entropy
        except ZeroDivisionError:
            return 1.0


class Completeness(metrics.base.MultiClassMetric):
    r"""Completeness Score.

    Completeness [^1] is symmetrical to homogeneity. In order to satisfy the
    completeness criteria, a clustering must assign all of those datapoints
    that are members of a single class to a single cluster. To evaluate completeness,
    we examine the distribution cluster assignments within each class. In a
    perfectly complete clustering solution, each of these distributions will be
    completely skewed to a single cluster.

    We can evaluate this degree of skew by calculating the conditional entropy of
    the proposed cluster distribution given the class of the component data points.
    However, in the worst case scenario, each class is represented by every cluster
    with a distribution equal to the distribution of cluster sizes. Therefore,
    symmetric to the claculation above, we define completeness as:

    $$
    c = \begin{cases}
    1 if H(K) = 0, \\
    1 - \frac{H(K|C)}{H(K)} otherwise.
    \end{cases}.
    $$

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

    >>> metric = metrics.Completeness()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    1.0
    0.3836885465963443
    0.5880325916843805
    0.6666666666666667

    >>> metric
    Completeness: 66.67%

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    @property
    def works_with_weights(self):
        return False

    def get(self):
        raw_conditional_entropy = 0
        max_reduction_entropy = 0

        for i in self.cm.classes:

            for j in self.cm.classes:
                try:
                    raw_conditional_entropy -= (
                        self.cm[i][j]
                        / self.cm.n_samples
                        * math.log(self.cm[i][j] / self.cm.sum_row[i])
                    )
                except (ValueError, ZeroDivisionError):
                    continue

            try:
                max_reduction_entropy -= (
                    self.cm.sum_col[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_col[i] / self.cm.n_samples)
                )
            except (ValueError, ZeroDivisionError):
                continue

        try:
            return 1.0 - raw_conditional_entropy / max_reduction_entropy
        except ZeroDivisionError:
            return 1.0


class VBeta(metrics.base.MultiClassMetric):
    r"""VBeta.

    VBeta (or V-Measure) [^1] is an external entropy-based cluster evaluation measure.
    It provides an elegant solution to many problems that affect previously defined
    cluster evaluation measures including

    * Dependance of clustering algorithm or dataset,

    * The "problem of matching", where the clustering of only a portion of data
    points are evaluated, and

    * Accurate evaluation and combination of two desirable aspects of clustering,
    homogeneity and completeness.

    Based upon the calculations of homogeneity and completeness, a clustering
    solution's V-measure is calculated by computing the weighted harmonic mean
    of homogeneity and completeness,

    $$
    V_{\beta} = \frac{(1 + \beta) \times h \times c}{\beta \times h + c}.
    $$

    Parameters
    ----------
    beta
        Weight of Homogeneity in the harmonic mean.
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.VBeta(beta=1.0)
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    0.0
    0.3437110184854507
    0.4580652856440158
    0.5158037429793888

    >>> metric
    VBeta: 51.58%

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    def __init__(self, beta: float = 1.0, cm=None):
        super().__init__(cm)
        self.beta = beta
        self.homogeneity = metrics.Homogeneity(self.cm)
        self.completeness = metrics.Completeness(self.cm)

    @property
    def works_with_weights(self):
        return False

    def get(self):
        h = self.homogeneity.get()
        c = self.completeness.get()

        try:
            return (1 + self.beta) * h * c / (self.beta * h + c)
        except ZeroDivisionError:
            return 0.0
