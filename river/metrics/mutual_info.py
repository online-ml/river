import math

import numpy as np
from scipy.special import factorial

from river import metrics

__all__ = [
    "AdjustedMutualInfo",
    "ExpectedMutualInfo",
    "MutualInfo",
    "NormalizedMutualInfo",
]


class MutualInfo(metrics.MultiClassMetric):
    r"""Mutual Information between two clusterings.

    The Mutual Information [^1] is a measure of the similarity between two labels of
    the same data. Where $|U_i|$ is the number of samples in cluster $U_i$ and $|V_j|$
    is the number of the samples in cluster $V_j$, the Mutual Information between
    clusterings $U$ and $V$ can be calculated as:

    $$
    MI(U,V) = \sum_{i=1}^{|U|} \sum_{v=1}^{|V|} \frac{|U_i \cup V_j|}{N} \log \frac{N |U_i \cup V_j|}{|U_i| |V_j|}
    $$

    This metric is independent of the absolute values of the labels: a permutation
    of the class or cluster label values won't change the score.

    This metric is furthermore symmetric: switching `y_true` and `y_pred` will return
    the same score value. This can be useful to measure the agreement of two independent
    label assignments strategies on the same dataset when the real ground truth is
    not known.

    The Mutual Information can be equivalently expressed as:

    $$
    MI(U,V) = H(U) - H(U | V) = H(V) - H(V | U)
    $$

    where $H(U)$ and $H(V)$ are the marginal entropies, $H(U | V)$ and $H(V | U)$ are the
    conditional entropies.

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

    >>> metric = metrics.MutualInfo()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.0
    0.21576155433883565
    0.39575279478527836
    0.46209812037329684

    >>> metric
    MutualInfo: 0.462098

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    [^2]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    """

    def __init__(self, cm=None):
        super().__init__(cm)

    @property
    def works_with_weights(self):
        return False

    def get(self):

        mutual_info_score = 0.0

        for i in self.cm.classes:
            for j in self.cm.classes:
                try:
                    mutual_info_score += (
                        self.cm[i][j]
                        / self.cm.n_samples
                        * math.log(
                            (self.cm.n_samples * self.cm[i][j])
                            / (self.cm.sum_row[i] * self.cm.sum_col[j])
                        )
                    )
                except (ValueError, ZeroDivisionError):
                    continue

        return mutual_info_score


class NormalizedMutualInfo(metrics.MultiClassMetric):
    r"""Normalized Mutual Information between two clusterings.

    Normalized Mutual Information (NMI) is a normalized version of the Mutual Information (MI) score
    to scale the results between the range of 0 (no mutual information) and 1 (perfectly mutual
    information). In the formula, the mutual information will be normalized by a generalized mean of
    the entropy of true and predicted labels, defined by the `average_method`.

    We note that this measure is not adjusted for chance (i.e corrected the effect of result
    agreement solely due to chance); as a result, the Adjusted Mutual Info Score will mostly be preferred.
    However, this metric is still symmetric, which means that switching true and predicted labels will not
    alter the score value. This fact can be useful when the metric is used to measure the agreement between
    two indepedent label solutions on the same dataset, when the ground truth remains unknown.

    Another advantage of the metric is that as it is based on the calculation of entropy-related measures,
    it is independent of the permutation of class/cluster labels.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    average_method
        This parameter defines how to compute the normalizer in the denominator.
        Possible options include `min`, `max`, `arithmetic` and `geometric`.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.NormalizedMutualInfo()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    0.0
    0.3437110184854508
    0.4580652856440159
    0.5158037429793888

    >>> metric
    NormalizedMutualInfo: 0.515804

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    [^2]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    """

    def __init__(self, cm=None, average_method="arithmetic"):
        super().__init__(cm)
        self.average_method = average_method

    @property
    def works_with_weights(self):
        return False

    @staticmethod
    def _generalized_average(u, v, average_method):

        if average_method == "min":
            return min(u, v)
        elif average_method == "max":
            return max(u, v)
        elif average_method == "geometric":
            return math.sqrt(u * v)
        elif average_method == "arithmetic":
            return (u + v) / 2
        else:
            raise ValueError(
                "'average_method' must be either 'min', 'max', "
                "'geometric', or 'arithmetic' "
            )

    def get(self):

        n_classes = len([i for i in self.cm.sum_row.values() if i != 0])
        n_clusters = len([i for i in self.cm.sum_col.values() if i != 0])

        if (n_classes == n_clusters == 1) or (n_classes == n_clusters == 0):
            return 1.0

        mutual_info_score = metrics.MutualInfo(self.cm).get()

        entropy_true = entropy_pred = 0.0

        for i in self.cm.classes:

            try:
                entropy_true -= (
                    self.cm.sum_row[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_row[i] / self.cm.n_samples)
                )
            except ValueError:
                pass

            try:
                entropy_pred -= (
                    self.cm.sum_col[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_col[i] / self.cm.n_samples)
                )
            except ValueError:
                pass

        normalizer = self._generalized_average(
            entropy_true, entropy_pred, self.average_method
        )

        normalizer = max(normalizer, np.finfo("float64").eps)

        return mutual_info_score / normalizer


class ExpectedMutualInfo(metrics.MultiClassMetric):
    r"""Expected Mutual Information.

    Like the Rand index, the baseline value of mutual information between two
    random clusterings is not necessarily a constant value, and tends to become
    larger when the two partitions have a higher number of cluster (with a fixed
    number of samples $N$). Using a hypergeometric model of randomness, it can be
    shown that the expected mutual information between two random clusterings [^1] is:

    $$
    E(MI(U, V)) = \sum_{i=1}^R \sum_{i=1}^C \sum{n_{ij}=(a_i+b_j-N)^+}^{\min(a_i,b_j)} \frac{n_{ij}}{N} \log(\frac{N n_{ij}}{a_i b_j}) \frac{a_i! b_j! (N-a_i)! (N-b_j)!}{N! n_{ij}! (a_i - n_{ij})! (b_j - n_{ij})! (N - a_i - b_j + n_{ij})!}}
    $$

    where

    * $(a_i + b_j - N)^+$ denotes $\max(1, a_i + b_j - N)$,

    * $a_i$ is the sum of row i-th of the contingency table, and

    * $b_j$ is the sum of column j-th of the contingency table.

    The Adjusted Mutual Information score (AMI) relies on the Expected Mutual Information (EMI) score.

    From the formula, we note that this metric is very expensive to calculate. As such,
    the AMI will be one order of magnitude slower than most other implemented metrics.

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

    >>> metric = metrics.ExpectedMutualInfo()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    0.0
    0.0
    0.21576155433883565
    0.34030102034048276
    0.2772588722239781

    >>> metric
    ExpectedMutualInfo: 0.277259

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    [^2]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    """

    def __init__(self, cm=None):
        super().__init__(cm)

    @property
    def works_with_weights(self):
        return False

    def get(self):

        expected_mutual_info = 0.0

        for i in self.cm.classes:
            for j in self.cm.classes:
                lower_bound = int(
                    max(1, self.cm.sum_row[i] + self.cm.sum_col[j] - self.cm.n_samples)
                )
                upper_bound = int(min(self.cm.sum_row[i], self.cm.sum_col[j]) + 1)

                for n_ij in range(lower_bound, upper_bound):
                    try:
                        expected_mutual_info += (
                            n_ij
                            / self.cm.n_samples
                            * math.log(
                                self.cm.n_samples
                                * n_ij
                                / (self.cm.sum_row[i] * self.cm.sum_col[j])
                            )
                            * (
                                factorial(self.cm.sum_row[i])
                                * factorial(self.cm.sum_col[j])
                                * factorial(self.cm.n_samples - self.cm.sum_row[i])
                                * factorial(self.cm.n_samples - self.cm.sum_col[j])
                            )
                            / (
                                factorial(self.cm.n_samples)
                                * factorial(n_ij)
                                * factorial(self.cm.sum_row[i] - n_ij)
                                * factorial(self.cm.sum_col[j] - n_ij)
                                * factorial(
                                    self.cm.n_samples
                                    - self.cm.sum_row[i]
                                    - self.cm.sum_col[j]
                                    + n_ij
                                )
                            )
                        )
                    except (ValueError, ZeroDivisionError):
                        continue

        return expected_mutual_info


class AdjustedMutualInfo(metrics.MultiClassMetric):
    r"""Adjusted Mutual Information between two clusterings.

    Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information score
    that accounts for chance. It corrects the effect of agreement solely due to chance
    between clusterings, similar to the way the Adjusted Rand Index corrects the Rand Index.
    It is closely related to variation of information. The adjusted measure, however, is
    no longer metrical.

    For two clusterings $U$ and $V$, the Adjusted Mutual Information is calculated as:

    $$
    AMI(U, V) = \frac{MI(U, V) - E(MI(U, V))}{avg(H(U), H(V)) - E(MI(U, V))}
    $$

    This metric is independent of the permutation of the class or cluster label values;
    furthermore, it is also symmetric. This can be useful to measure the agreement of
    two label assignments strategies on the same dataset, regardless of the ground truth.

    However, due to the complexity of the Expected Mutual Info Score, the computation of
    this metric is an order of magnitude slower than most other metrics, in general.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    average_method
        This parameter defines how to compute the normalizer in the denominator.
        Possible options include `min`, `max`, `arithmetic` and `geometric`.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.AdjustedMutualInfo()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    1.0
    1.0
    0.0
    0.0
    0.10589171576292913
    0.29879245817089006

    >>> metric
    AdjustedMutualInfo: 0.298792

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    [^2]: Andrew Rosenberg and Julia Hirschberg (2007).
          V-Measure: A conditional entropy-based external cluster evaluation measure.
          Proceedings of the 2007 Joing Conference on Empirical Methods in Natural Language
          Processing and Computational Natural Language Learning, pp. 410 - 420,
          Prague, June 2007.
    """

    def __init__(self, cm=None, average_method="arithmetic"):
        super().__init__(cm)
        self.average_method = average_method

    @property
    def works_with_weights(self):
        return False

    @staticmethod
    def _generalized_average(u, v, average_method):

        if average_method == "min":
            return min(u, v)
        elif average_method == "max":
            return max(u, v)
        elif average_method == "geometric":
            return math.sqrt(u * v)
        elif average_method == "arithmetic":
            return (u + v) / 2
        else:
            raise ValueError(
                "'average_method' must be either 'min', 'max', "
                "'geometric', or 'arithmetic' "
            )

    def get(self):

        n_classes = len([i for i in self.cm.sum_row.values() if i != 0])
        n_clusters = len([i for i in self.cm.sum_col.values() if i != 0])

        if (n_classes == n_clusters == 1) or (n_classes == n_clusters == 0):
            return 1.0

        mutual_info_score = metrics.MutualInfo(self.cm).get()

        expected_mutual_info_score = metrics.ExpectedMutualInfo(self.cm).get()

        entropy_true = entropy_pred = 0.0

        for i in self.cm.classes:

            try:
                entropy_true -= (
                    self.cm.sum_row[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_row[i] / self.cm.n_samples)
                )
            except ValueError:
                pass

            try:
                entropy_pred -= (
                    self.cm.sum_col[i]
                    / self.cm.n_samples
                    * math.log(self.cm.sum_col[i] / self.cm.n_samples)
                )
            except ValueError:
                pass

        normalizer = self._generalized_average(
            entropy_true, entropy_pred, self.average_method
        )

        denominator = normalizer - expected_mutual_info_score

        if denominator > 0:
            denominator = max(denominator, np.finfo("float64").eps)
        else:
            denominator = -min(denominator, -np.finfo("float64").eps)

        return (mutual_info_score - expected_mutual_info_score) / denominator
