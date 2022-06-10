import math

import numpy as np

from river import metrics
from river.metrics.expected_mutual_info import expected_mutual_info

__all__ = [
    "AdjustedMutualInfo",
    "MutualInfo",
    "NormalizedMutualInfo",
]


class MutualInfo(metrics.base.MultiClassMetric):
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
    0.215761
    0.395752
    0.462098

    >>> metric
    MutualInfo: 0.462098

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    """

    _fmt = ""

    @property
    def works_with_weights(self):
        return False

    def get(self):

        mutual_info_score = 0.0

        for i in self.cm.classes:
            for j in self.cm.classes:
                try:
                    temp = (
                        self.cm[i][j]
                        / self.cm.n_samples
                        * (
                            math.log(self.cm.n_samples * self.cm[i][j])
                            - math.log(self.cm.sum_row[i] * self.cm.sum_col[j])
                        )
                    )
                except (ValueError, ZeroDivisionError):
                    continue
                temp = 0.0 if (abs(temp) < np.finfo("float64").eps) else temp
                # temp = 0.0 if temp < 0.0 else temp   # TODO confirm if we need to clip here
                mutual_info_score += temp
        return mutual_info_score


class NormalizedMutualInfo(metrics.base.MultiClassMetric):
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
    0.343711
    0.458065
    0.515803

    >>> metric
    NormalizedMutualInfo: 0.515804

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    """
    _AVERAGE_MIN = "min"
    _AVERAGE_MAX = "max"
    _AVERAGE_GEOMETRIC = "geometric"
    _AVERAGE_ARITHMETIC = "arithmetic"
    _VALID_AVERAGE = [
        _AVERAGE_MIN,
        _AVERAGE_MAX,
        _AVERAGE_GEOMETRIC,
        _AVERAGE_ARITHMETIC,
    ]

    _fmt = ""

    def __init__(self, cm=None, average_method="arithmetic"):
        super().__init__(cm)
        if average_method not in self._VALID_AVERAGE:
            raise ValueError(
                f"Valid 'average_methods' are {self._VALID_AVERAGE}, "
                f"but {average_method} was passed."
            )
        self.average_method = average_method
        if average_method == self._AVERAGE_MIN:
            self._generalized_average = min
        elif average_method == self._AVERAGE_MAX:
            self._generalized_average = max
        elif average_method == self._AVERAGE_GEOMETRIC:
            self._generalized_average = _average_geometric
        else:  # average_method == self._AVERAGE_ARITHMETIC
            self._generalized_average = _average_arithmetic

    @property
    def works_with_weights(self):
        return False

    def get(self):

        n_classes = len([i for i in self.cm.sum_row.values() if i != 0])
        n_clusters = len([i for i in self.cm.sum_col.values() if i != 0])

        if (n_classes == n_clusters == 1) or (n_classes == n_clusters == 0):
            return 1.0

        mutual_info_score = metrics.MutualInfo(self.cm).get()

        entropy_true = _entropy(cm=self.cm, y_true=True)
        entropy_pred = _entropy(cm=self.cm, y_true=False)

        normalizer = self._generalized_average(entropy_true, entropy_pred)

        normalizer = max(normalizer, np.finfo("float64").eps)

        return mutual_info_score / normalizer


class AdjustedMutualInfo(metrics.base.MultiClassMetric):
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
    0.105891
    0.298792

    >>> metric
    AdjustedMutualInfo: 0.298792

    References
    ----------
    [^1]: Wikipedia contributors. (2021, March 17). Mutual information.
          In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Mutual_information&oldid=1012714929
    """
    _AVERAGE_MIN = "min"
    _AVERAGE_MAX = "max"
    _AVERAGE_GEOMETRIC = "geometric"
    _AVERAGE_ARITHMETIC = "arithmetic"
    _VALID_AVERAGE = [
        _AVERAGE_MIN,
        _AVERAGE_MAX,
        _AVERAGE_GEOMETRIC,
        _AVERAGE_ARITHMETIC,
    ]

    _fmt = ""

    def __init__(self, cm=None, average_method="arithmetic"):
        super().__init__(cm)
        self.average_method = average_method
        if average_method not in self._VALID_AVERAGE:
            raise ValueError(
                f"Valid 'average_methods' are {self._VALID_AVERAGE}, "
                f"but {average_method} was passed."
            )
        self.average_method = average_method
        if average_method == self._AVERAGE_MIN:
            self._generalized_average = min
        elif average_method == self._AVERAGE_MAX:
            self._generalized_average = max
        elif average_method == self._AVERAGE_GEOMETRIC:
            self._generalized_average = _average_geometric
        else:  # average_method == self._AVERAGE_ARITHMETIC
            self._generalized_average = _average_arithmetic

    @property
    def works_with_weights(self):
        return False

    def get(self):

        n_classes = len([i for i in self.cm.sum_row.values() if i != 0])
        n_clusters = len([i for i in self.cm.sum_col.values() if i != 0])

        if (n_classes == n_clusters == 1) or (n_classes == n_clusters == 0):
            return 1.0

        mutual_info_score = metrics.MutualInfo(self.cm).get()

        expected_mutual_info_score = expected_mutual_info(self.cm)

        entropy_true = _entropy(cm=self.cm, y_true=True)
        entropy_pred = _entropy(cm=self.cm, y_true=False)

        normalizer = self._generalized_average(entropy_true, entropy_pred)

        denominator = normalizer - expected_mutual_info_score

        if denominator < 0:
            denominator = min(denominator, -np.finfo("float64").eps)
        else:
            denominator = max(denominator, np.finfo("float64").eps)

        adjusted_mutual_info_score = (mutual_info_score - expected_mutual_info_score) / denominator

        return adjusted_mutual_info_score


def _entropy(cm, y_true):
    n_samples = cm.n_samples
    if n_samples == 0:
        return 1.0

    if y_true:
        values = cm.sum_row
    else:
        values = cm.sum_col
    entropy = 0.0
    for i in cm.classes:
        if i in values and values[i] > 0:
            entropy -= (values[i] / n_samples) * (np.log(values[i]) - np.log(n_samples))
    return entropy


def _average_geometric(u, v):
    return math.sqrt(u * v)


def _average_arithmetic(u, v):
    return (u + v) / 2
