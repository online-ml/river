import collections
import statistics

from .. import stats

from . import base


__all__ = [
    'MacroPrecision',
    'MicroPrecision',
    'Precision'
]


class BasePrecision:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class Precision(stats.Mean, BasePrecision, base.BinaryMetric):
    """Binary precision score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.Precision()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            Precision: 1.
            Precision: 0.5
            Precision: 0.5
            Precision: 0.666667
            Precision: 0.75

    """

    def update(self, y_true, y_pred, sample_weight=1.):
        if y_pred:
            super().update(x=y_true == y_pred, w=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        if y_pred:
            super().revert(x=y_true == y_pred, w=sample_weight)
        return self


class MacroPrecision(BasePrecision, base.MultiClassMetric):
    """Macro-average precision score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroPrecision()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroPrecision: 1.
            MacroPrecision: 0.25
            MacroPrecision: 0.5
            MacroPrecision: 0.5
            MacroPrecision: 0.5

    """

    def __init__(self):
        self.precisions = collections.defaultdict(Precision)
        self._class_counts = collections.Counter()

    def update(self, y_true, y_pred, sample_weight=1.):
        self.precisions[y_pred].update(y_true == y_pred, True, sample_weight)
        self._class_counts.update([y_true, y_pred])
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        self.precisions[y_pred].revert(y_true == y_pred, True, sample_weight)
        self._class_counts.subtract([y_true, y_pred])
        return self

    def get(self):
        if not self._class_counts:
            return 0.
        return statistics.mean((
            0. if c not in self.precisions else self.precisions[c].get()
            for c, count in self._class_counts.items()
            if count > 0
        ))


class MicroPrecision(stats.Mean, BasePrecision, base.MultiClassMetric):
    """Micro-average precision score.

    The micro-average precision score is exactly equivalent to the micro-average recall as well as
    the micro-average F1 score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroPrecision()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MicroPrecision: 1.
            MicroPrecision: 0.5
            MicroPrecision: 0.666667
            MicroPrecision: 0.75
            MicroPrecision: 0.6

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    def update(self, y_true, y_pred, sample_weight=1.):
        super().update(x=y_true == y_pred, w=sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight=1.):
        super().revert(x=y_true == y_pred, w=sample_weight)
        return self
