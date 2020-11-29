from . import base
from . import precision


__all__ = ["MacroRecall", "MicroRecall", "Recall", "WeightedRecall", "ExampleRecall"]


class Recall(base.BinaryMetric):
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
    Recall: 1.
    Recall: 1.
    Recall: 0.5
    Recall: 0.666667
    Recall: 0.75

    """

    def get(self):
        tp = self.cm.true_positives(self.pos_val)
        fn = self.cm.false_negatives(self.pos_val)
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.0


class MacroRecall(base.MultiClassMetric):
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
            return 0.0


class MicroRecall(precision.MicroPrecision):
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
    MicroRecall: 1.
    MicroRecall: 0.5
    MicroRecall: 0.666667
    MicroRecall: 0.75
    MicroRecall: 0.6

    References
    ----------
    [^1]: [Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem?](https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/)

    """


class WeightedRecall(base.MultiClassMetric):
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
            return total / self.cm.total_weight
        except ZeroDivisionError:
            return 0.0


class ExampleRecall(base.MultiOutputClassificationMetric):
    """Example-based recall score for multilabel classification.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion matrix between multiple metrics. Sharing a
        confusion matrix reduces the amount of storage and computation time.

    Examples
    --------

    >>> from river import metrics

    >>> y_true = [
    ...     {0: False, 1: True, 2: True},
    ...     {0: True, 1: True, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> y_pred = [
    ...     {0: True, 1: True, 2: True},
    ...     {0: True, 1: False, 2: False},
    ...     {0: True, 1: True, 2: False},
    ... ]

    >>> metric = metrics.ExampleRecall()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    ExampleRecall: 0.833333

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def get(self):

        try:
            return self.cm.recall_sum / self.cm.n_samples
        except ZeroDivisionError:
            return 0.0
