import math

from river import metrics

__all__ = ["FowlkesMallows"]


class FowlkesMallows(metrics.base.MultiClassMetric):
    r"""Fowlkes-Mallows Index.

    The Fowlkes-Mallows Index [^1] [^2] is an external evaluation method that is
    used to determine the similarity between two clusterings, and also a metric
    to measure confusion matrices. The measure of similarity could be either between
    two hierarchical clusterings or a clustering and a benchmark classification. A
    higher value for the Fowlkes-Mallows index indicates a greater similarity between
    the clusters and the benchmark classifications.

    The Fowlkes-Mallows Index, for two cluster algorithms, is defined as:

    $$
    FM = \sqrt{PPV \times TPR} = \sqrt{\frac{TP}{TP+FP} \times \frac{TP}{TP+FN}}
    $$

    where

    * TP, FP, FN are respectively the number of true positives, false positives and
    false negatives;

    * TPR is the True Positive Rate (or Sensitivity/Recall), PPV is the Positive Predictive
    Rate (or Precision).

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [0, 0, 0, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 1, 2, 2]

    >>> metric = metrics.FowlkesMallows()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    FowlkesMallows: 0.00%
    FowlkesMallows: 100.00%
    FowlkesMallows: 57.74%
    FowlkesMallows: 40.82%
    FowlkesMallows: 35.36%
    FowlkesMallows: 47.14%

    References
    ----------
    [^1]: Wikipedia contributors. (2020, December 22).
          Fowlkes–Mallows index. In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Fowlkes%E2%80%93Mallows_index&oldid=995714222
    [^2]: E. B. Fowkles and C. L. Mallows (1983).
          “A method for comparing two hierarchical clusterings”.
          Journal of the American Statistical Association

    """

    @property
    def works_with_weights(self):
        return False

    def get(self):

        n = self.cm.n_samples
        tk = sum(c * c for row in self.cm.data.values() for c in row.values()) - n
        pk = sum(sc * sc for sc in self.cm.sum_col.values()) - n
        qk = sum(sr * sr for sr in self.cm.sum_row.values()) - n

        try:
            return math.sqrt(tk / pk) * math.sqrt(tk / qk)
        except ZeroDivisionError:
            return 0.0
