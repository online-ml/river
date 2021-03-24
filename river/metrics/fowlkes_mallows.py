import math

from river import metrics, utils

from . import base

__all__ = ["FowlkesMallows", "PPV", "TPR"]


class PPV(base.Metric):
    r"""Positive Predictive Rate (PPV).

    If we define

    * True Positive (TP) as the number of pairs of points that are present in the
    same cluster in both true and predicted labels.

    * False Positive (FP) as the number of pairs of points that are present in the
    same cluster in labels but not in predicted labels.

    * False Negative (FN) as the number of pairs of points that are present in the
    same cluster in predicted labels but not in true labels.

    * False Positive (FP) as the number of pairs of points that are in different
     clusters in both true and predicted labels.

    The Positive Predictive Rate - PPV [^1] [^2] (or Precision) can be calculated as

    $$
    PPV = \frac{TP}{TP + FP}
    $$

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.PPV()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    1.0
    1.0
    0.5
    0.5
    0.6666666666666666

    >>> metric
    PPV: 0.666667

    References
    ----------
    [^1]: Wikipedia contributors. (2020, December 22).
          Fowlkes–Mallows index. In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Fowlkes%E2%80%93Mallows_index&oldid=995714222
    [^2]: E. B. Fowkles and C. L. Mallows (1983).
          “A method for comparing two hierarchical clusterings”.
          Journal of the American Statistical Association

    """

    def __init__(self):
        super().__init__()

        self.cm = metrics.ConfusionMatrix()
        self._true_positive = 0
        self._false_positive = 0

    def update(self, y_true, y_pred, sample_weight=1.0):

        self.cm.update(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 2:
            self._true_positive += self.cm[y_true][y_pred] - 1

        self._false_positive += self.cm.sum_row[y_true] - self.cm[y_true][y_pred]

        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):

        self.cm.revert(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 1:
            self._true_positive -= self.cm[y_true][y_pred]

        self._false_positive -= self.cm.sum_row[y_true] - self.cm[y_true][y_pred]

        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)

    def get(self):

        try:
            return self._true_positive / (self._true_positive + self._false_positive)
        except ZeroDivisionError:
            return 0.0


class TPR(base.Metric):
    r"""True Predictive Rate (TPR).

    If we define

    * True Positive (TP) as the number of pairs of points that are present in the
    same cluster in both true and predicted labels.

    * False Positive (FP) as the number of pairs of points that are present in the
    same cluster in labels but not in predicted labels.

    * False Negative (FN) as the number of pairs of points that are present in the
    same cluster in predicted labels but not in true labels.

    * False Positive (FP) as the number of pairs of points that are in different
     clusters in both true and predicted labels.

    The True Positive Rate - TPR [^1] [^2] (or Sensiivity or Recall) can be calculated as

    $$
    PPV = \frac{TP}{TP + FN}
    $$

    Examples
    --------
    >>> from river import metrics

    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 2, 2]

    >>> metric = metrics.TPR()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp).get())
    0.0
    1.0
    0.3333333333333333
    0.3333333333333333
    0.25
    0.3333333333333333

    >>> metric
    TPR: 0.333333

    References
    ----------
    [^1]: Wikipedia contributors. (2020, December 22).
          Fowlkes–Mallows index. In Wikipedia, The Free Encyclopedia,
          from https://en.wikipedia.org/w/index.php?title=Fowlkes%E2%80%93Mallows_index&oldid=995714222
    [^2]: E. B. Fowkles and C. L. Mallows (1983).
          “A method for comparing two hierarchical clusterings”.
          Journal of the American Statistical Association

    """

    def __init__(self):
        super().__init__()

        self.cm = metrics.ConfusionMatrix()
        self._true_positive = 0
        self._false_negative = 0

    def update(self, y_true, y_pred, sample_weight=1.0):

        self.cm.update(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 2:
            self._true_positive += self.cm[y_true][y_pred] - 1

        self._false_negative += self.cm.sum_col[y_pred] - self.cm[y_true][y_pred]

        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):

        self.cm.revert(y_true, y_pred)

        if self.cm[y_true][y_pred] >= 1:
            self._true_positive -= self.cm[y_true][y_pred]

        self._false_negative -= self.cm.sum_col[y_pred] - self.cm[y_true][y_pred]

        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)

    def get(self):

        try:
            return self._true_positive / (self._true_positive + self._false_negative)
        except ZeroDivisionError:
            return 0.0


class FowlkesMallows(base.Metric):
    r"""Fowlkes-Mallows Index.

    The Fowlkes-Mallows Index [^1] [^2] is an external evaluation method that is
    used to determine the similarity between two clusterings, and also a mmetric
    to measure confusion matrices. The measure of similarity could be either between
    two hierarchical clusterings or a clustering and a benchmark classification. A
    higher value for teh Fowlkes-Mallows index indicates a greater similarity between
    the clusters and the benchmark classifications.

    The Fowlkes-Mallows Index, when results of two cluster algorithms are used to
    evaluate the results, is defined as:

    $$
    FM = \sqrt{PPV \times TPR} = \sqrt{\frac{TP}{TP+FP} \times \frac{TP}{TP+FN}}
    $$

    where

    * TP, FP, FN are respectively the number of true positives, false positives and
    false negatives;

    * TPR is the True Positive Rate (or Sensitivity/Recall), PPV is the Positive Predictive
    Rate (or Precision).

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

    def __init__(self):
        super().__init__()
        self.ppv = metrics.PPV()
        self.tpr = metrics.TPR()

    def update(self, y_true, y_pred, sample_weight=1.0):

        self.ppv.update(y_true, y_pred)
        self.tpr.update(y_true, y_pred)

        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):

        self.ppv.revert(y_true, y_pred)
        self.tpr.revert(y_true, y_pred)

        return self

    @property
    def bigger_is_better(self):
        return True

    def works_with(self, model):
        return utils.inspect.isclassifier(model) or utils.inspect.isclusterer(model)

    def get(self):

        try:
            return math.sqrt(self.ppv.get() * self.tpr.get())
        except ValueError:
            return 0.0
