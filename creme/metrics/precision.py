import collections
import functools
import statistics

from .. import stats

from . import base


__all__ = [
    'MacroPrecision',
    'MicroPrecision',
    'Precision',
    'RollingMacroPrecision',
    'RollingMicroPrecision',
    'RollingPrecision'
]


class BasePrecision:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class Precision(stats.Mean, BasePrecision, base.BinaryClassificationMetric):
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

    def update(self, y_true, y_pred):
        if y_pred:
            return super().update(y_true == y_pred)
        return self


class RollingPrecision(stats.RollingMean, BasePrecision, base.BinaryClassificationMetric):
    """Rolling binary precision score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingPrecision(window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p))
            RollingPrecision: 1.
            RollingPrecision: 0.5
            RollingPrecision: 0.5
            RollingPrecision: 0.666667
            RollingPrecision: 0.666667

    """

    def update(self, y_true, y_pred):
        if y_pred:
            return super().update(y_true == y_pred)
        return self


class MacroPrecision(BasePrecision, base.MultiClassificationMetric):
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

    def update(self, y_true, y_pred):
        self.precisions[y_pred].update(y_true == y_pred, True)
        self.classes.update({y_true, y_pred})
        return self

    def get(self):
        return statistics.mean((
            0 if c not in self.precisions else self.precisions[c].get()
            for c in self.classes
        ))


class RollingMacroPrecision(MacroPrecision):
    """Rolling macro-average precision score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroPrecision(window_size=2)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            0.333333...
            0.125
            0.194444...
            0.25
            0.25

            >>> metric
            RollingMacroPrecision: 0.25

    """

    def __init__(self, window_size):
        self.precisions = collections.defaultdict(functools.partial(Precision, window_size))
        self.classes = set()


class MicroPrecision(stats.Mean, BasePrecision, base.MultiClassificationMetric):
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

    def update(self, y_true, y_pred):
        super().update(y_true == y_pred)
        return self


class RollingMicroPrecision(stats.RollingMean, BasePrecision, base.MultiClassificationMetric):
    """Rolling micro-average precision score.

    The micro-average precision score is exactly equivalent to the micro-average recall as well as
    the micro-average F1 score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import precision_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMicroPrecision(window_size=3)
            >>> for y_t, y_p in zip(y_true, y_pred):
            ...     print(metric.update(y_t, y_p).get())
            1.0
            0.5
            0.666666...
            0.666666...
            0.666666...

            >>> metric
            RollingMicroPrecision: 0.666667

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    def update(self, y_true, y_pred):
        super().update(y_true == y_pred)
        return self
