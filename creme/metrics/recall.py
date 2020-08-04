import collections

from . import base
from . import precision


__all__ = [
    'MacroRecall',
    'MicroRecall',
    'Recall',
    'WeightedRecall'
]


class Recall(base.BinaryMetric):
    """Binary recall score.

    Examples
    --------

    >>> from creme import metrics

    >>> y_true = [True, False, True, True, True]
    >>> y_pred = [True, True, False, True, True]

    >>> metric = metrics.Recall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    Recall: 1.
    Recall: 1.
    Recall: 0.5
    Recall: 0.666667
    Recall: 0.75

    """

    def get(self):
        tp = self.cm.true_positives
        fn = self.cm.false_negatives
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.


class MacroRecall(base.MultiClassMetric):
    """Macro-average recall score.

    Examples
    --------

    >>> from creme import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MacroRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MacroRecall: 1.
    MacroRecall: 0.5
    MacroRecall: 0.666667
    MacroRecall: 0.666667
    MacroRecall: 0.555556

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
            return 0.


class MicroRecall(precision.MicroPrecision):
    """Micro-average recall score.

    The micro-average recall is exactly equivalent to the micro-average precision as well as the
    micro-average F1 score.

    Examples
    --------

    >>> from creme import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.MicroRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    MicroRecall: 1.
    MicroRecall: 0.5
    MicroRecall: 0.666667
    MicroRecall: 0.75
    MicroRecall: 0.6

    References
    ----------

    1. [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """


class WeightedRecall(base.MultiClassMetric):
    """Weighted-average recall score.

    This uses the support of each label to compute an average score, whereas `MacroRecall`
    ignores the support.

    Examples
    --------

    >>> from creme import metrics

    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]

    >>> metric = metrics.WeightedRecall()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     print(metric.update(yt, yp))
    WeightedRecall: 1.
    WeightedRecall: 0.5
    WeightedRecall: 0.666667
    WeightedRecall: 0.75
    WeightedRecall: 0.6

    """

    def get(self):
        total = 0
        for c in self.cm.classes:
            try:
                total += self.cm[c][c]
            except ZeroDivisionError:
                continue
        try:
            return total / self.cm.n_samples
        except ZeroDivisionError:
            return 0.
