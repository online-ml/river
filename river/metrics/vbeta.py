import math

from river import metrics

__all__ = ["Completeness", "Homogeneity", "VBeta"]


class Homogeneity(metrics.MultiClassMetric):
    r"""Homogeneity Score.

    Homogeneity metric [^1] of a cluster labeling given a ground truth.

    In order to satisfy the homogeneity criteria, a clustering must assign only
    those data points that a members of a single class to a single cluster. That
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

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.Homogeneity()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Homogeneity: 1.
    Homogeneity: 1.
    Homogeneity: 0.
    Homogeneity: 0.311278
    Homogeneity: 0.37515
    Homogeneity: 0.42062

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    def __init__(self, cm=None):
        super().__init__(cm)

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


class Completeness(metrics.MultiClassMetric):
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

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.Completeness()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Completeness: 1.
    Completeness: 1.
    Completeness: 1.
    Completeness: 0.383689
    Completeness: 0.588033
    Completeness: 0.666667

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    def __init__(self, cm=None):
        super().__init__(cm)

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


class VBeta(metrics.MultiClassMetric):
    """V-Meeasure.

    V-measure [^1] is an external entropy-based cluster evaluation measure. It provides
    an elegant solution to many problems that affect previously defined cluster
    evaluation measures including

    * Dependence of clustering algorithm or dataset,

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

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.VBeta(beta=1.0)
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    VBeta: 1.
    VBeta: 1.
    VBeta: 0.
    VBeta: 0.343711
    VBeta: 0.458065
    VBeta: 0.515804

    References
    ----------
    [^1]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.

    """

    def __init__(self, beta: float, cm=None):
        super().__init__(cm)
        self.beta = beta
        self.homogeneity = metrics.Homogeneity(self.cm)
        self.completeness = metrics.Completeness(self.cm)

    def get(self):
        h = self.homogeneity.get()
        c = self.completeness.get()

        try:
            return (1 + self.beta) * h * c / (self.beta * h + c)
        except ZeroDivisionError:
            return 0.0
