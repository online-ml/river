from __future__ import annotations

from river import metrics

__all__ = ["MacroRecall", "MicroRecall", "Recall", "WeightedRecall"]


class Recall(metrics.base.BinaryMetric):
    """Binary recall score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.
    pos_val
        Value to treat as "positive".

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [True, False, True, True, True]
    >>> y_pred = [True, True, False, True, True]

    >>> metric = metrics.Recall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Recall: 100.00%
    Recall: 100.00%
    Recall: 50.00%
    Recall: 66.67%
    Recall: 75.00%

    """

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fn = self.cm.false_negatives(self.pos_val)
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.0


class MacroRecall(metrics.base.MultiClassMetric):
    """Macro-average recall score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MacroRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroRecall: 100.00%
    MacroRecall: 50.00%
    MacroRecall: 66.67%
    MacroRecall: 66.67%
    MacroRecall: 55.56%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c] / self.cm.sum_row[c]
            except ZeroDivisionError:
                continue
        try:
            return total / len(self.cm.classes)
        except ZeroDivisionError:
            return 0.0


class MicroRecall(metrics.MicroPrecision):
    """Micro-average recall score.

    The micro-average recall is exactly equivalent to the micro-average precision as well as the
    micro-average F1 score.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MicroRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MicroRecall: 100.00%
    MicroRecall: 50.00%
    MicroRecall: 66.67%
    MicroRecall: 75.00%
    MicroRecall: 60.00%

    References
    ----------
    [^1]: [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """


class WeightedRecall(metrics.base.MultiClassMetric):
    """Weighted-average recall score.

    This uses the support of each label to compute an average score, whereas `MacroRecall`
    ignores the support.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.WeightedRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedRecall: 100.00%
    WeightedRecall: 50.00%
    WeightedRecall: 66.67%
    WeightedRecall: 75.00%
    WeightedRecall: 60.00%

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c]
            except ZeroDivisionError:
                continue
        try:
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0
