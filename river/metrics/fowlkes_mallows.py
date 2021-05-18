import math

from river import metrics

from . import base

__all__ = ["FowlkesMallows"]


class FowlkesMallows(base.MultiClassMetric):
    r"""Fowlkes-Mallows Index.

    The Fowlkes-Mallows Index [^1] [^2] is an external evaluation method that is
    used to determine the similarity between two clusterings, and also a mmetric
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

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.FowlkesMallows()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    1.0
    0.5773502691896257
    0.408248290463863
    0.3535533905932738
    0.4714045207910317

    >>> metric
    FowlkesMallows: 0.471405

    References
    ----------
    [^1]: Wikipedia contributors. (2020, December 22).
          Fowlkes–Mallows index. In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Fowlkes%E2%80%93Mallows_index&oldid=995714222
    [^2]: E. B. Fowkles and C. L. Mallows (1983).
          “A method for comparing two hierarchical clusterings”.
          Journal of the American Statistical Association

    """

    def __init__(self, cm=None):
        super().__init__(cm)

    @property
    def works_with_weights(self):
        return False

    def get(self):

        pair_confusion_matrix = metrics.PairConfusionMatrix(self.cm).get()

        true_positives = pair_confusion_matrix[1][1]
        false_positives = pair_confusion_matrix[0][1]
        false_negatives = pair_confusion_matrix[1][0]

        try:
            ppv = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            ppv = 0.0

        try:
            tpr = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            tpr = 0.0

        return math.sqrt(ppv * tpr)
