import collections
import statistics

from .. import stats

from . import base
from . import precision


__all__ = ['Recall', 'MacroRecall', 'MicroRecall']


class Recall(stats.Mean, base.BinaryClassificationMetric):
    """Binary recall score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import recall_score

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.Recall()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert metric.get() == recall_score(y_true[:i+1], y_pred[:i+1])

            >>> metric
            Recall: 0.75

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        if y_true:
            return super().update(y_true == y_pred)
        return self


class MacroRecall(base.MultiClassificationMetric):
    """Macro-average recall score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import recall_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroRecall()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), recall_score(y_true[:i+1], y_pred[:i+1], average='macro'))
            1.0 1.0
            0.5 0.5
            0.666666... 0.666666...
            0.666666... 0.666666...
            0.555555... 0.555555...

            >>> metric
            MacroRecall: 0.555556

    """

    def __init__(self):
        self.recalls = collections.defaultdict(Recall)
        self.classes = set()

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        self.recalls[y_true].update(True, y_true == y_pred)
        self.classes.update({y_true, y_pred})
        return self

    def get(self):
        return statistics.mean((
            0 if c not in self.recalls else self.recalls[c].get()
            for c in self.classes
        ))


class MicroRecall(precision.MicroPrecision):
    """Micro-average recall score.

    The micro-average recall is exactly equivalent to the micro-average precision as well as the
    micro-average F1 score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import recall_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroRecall()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), recall_score(y_true[:i+1], y_pred[:i+1], average='micro'))
            1.0 1.0
            0.5 0.5
            0.666666... 0.666666...
            0.75 0.75
            0.6 0.6

            >>> metric
            MicroRecall: 0.6

    References:

        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """
