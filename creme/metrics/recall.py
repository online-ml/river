import collections

from .. import stats

from . import base
from . import precision


__all__ = [
    'MacroRecall',
    'MicroRecall',
    'Recall',
    'WeightedRecall'
]


class BaseRecall:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class Recall(stats.Mean, BaseRecall, base.BinaryMetric):
    """Binary recall score.

    Example:

        ::

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

    def update(self, y_true, y_pred, sample_weight=1.):
        if y_true:
            return super().update(x=y_true == y_pred, w=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        if y_true:
            return super().revert(x=y_true == y_pred, w=sample_weight)
        return self


class MacroRecall(BaseRecall, base.MultiClassMetric):
    """Macro-average recall score.

    Example:

        ::

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

    def __init__(self):
        self.recalls = collections.defaultdict(Recall)
        self._class_counts = collections.Counter()

    def update(self, y_true, y_pred, sample_weight=1.):
        self.recalls[y_true].update(True, y_true == y_pred, sample_weight)
        self._class_counts.update([y_true, y_pred])
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.recalls[y_true].revert(True, y_true == y_pred, sample_weight)
        self._class_counts.subtract([y_true, y_pred])
        return self

    def get(self):
        relevant = [c for c, count in self._class_counts.items() if count > 0]
        try:
            return sum(self.recalls[c].get() for c in relevant) / len(relevant)
        except ZeroDivisionError:
            return 0.


class MicroRecall(precision.MicroPrecision):
    """Micro-average recall score.

    The micro-average recall is exactly equivalent to the micro-average precision as well as the
    micro-average F1 score.

    Example:

        ::

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

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """


class WeightedRecall(BaseRecall, base.MultiClassMetric):
    """Weighted-average recall score.

    This uses the support of each label to compute an average score, whereas `MacroRecall` ignores
    the support.

    Example:

        ::

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

    def __init__(self):
        self.recalls = collections.defaultdict(Recall)
        self.support = collections.Counter()
        self._class_counts = collections.Counter()

    def update(self, y_true, y_pred, sample_weight=1.):
        self.recalls[y_true].update(True, y_true == y_pred, sample_weight)
        self.support.update({y_true: sample_weight})
        self._class_counts.update([y_true, y_pred])
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.recalls[y_true].revert(True, y_true == y_pred, sample_weight)
        self.support.subtract({y_true: sample_weight})
        self._class_counts.subtract([y_true, y_pred])
        return self

    def get(self):
        relevant = [c for c, count in self._class_counts.items() if count > 0]
        try:
            return (
                sum(self.recalls[c].get() * self.support[c] for c in relevant) /
                sum(self.support[c] for c in relevant)
            )
        except ZeroDivisionError:
            return 0.
