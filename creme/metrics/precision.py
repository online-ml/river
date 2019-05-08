import collections
import statistics

from .. import stats

from . import base


__all__ = ['Precision', 'MacroPrecision', 'MicroPrecision']


class Precision(stats.Mean, base.BinaryClassificationMetric):
    """Binary precision score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.Precision()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert metric.get() == precision_score(y_true[:i+1], y_pred[:i+1])

            >>> metric
            Precision: 0.75

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        if y_pred:
            return super().update(y_true == y_pred)
        return self


class MacroPrecision(base.MultiClassificationMetric):
    """Macro-average precision score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroPrecision()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), precision_score(y_true[:i+1], y_pred[:i+1], average='macro'))
            1.0 1.0
            0.25 0.25
            0.5 0.5
            0.5 0.5
            0.5 0.5

            >>> metric
            MacroPrecision: 0.5

    """

    def __init__(self):
        self.precisions = collections.defaultdict(Precision)
        self.classes = set()

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        self.precisions[y_pred].update(y_true == y_pred, True)
        self.classes.update({y_true, y_pred})
        return self

    def get(self):
        return statistics.mean((
            0 if c not in self.precisions else self.precisions[c].get()
            for c in self.classes
        ))


class MicroPrecision(stats.Mean, base.MultiClassificationMetric):
    """Micro-average precision score.

    The micro-average precision score is exactly equivalent to the micro-average recall as well as
    the micro-average F1 score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroPrecision()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), precision_score(y_true[:i+1], y_pred[:i+1], average='micro'))
            1.0 1.0
            0.5 0.5
            0.666666... 0.666666...
            0.75 0.75
            0.6 0.6

            >>> metric
            MicroPrecision: 0.6

    References:

        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        super().update(y_true == y_pred)
        return self
