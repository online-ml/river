import collections
import itertools
import statistics

from .. import stats

from . import base
from . import confusion


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

    def update(self, y_true, y_pred):
        if y_pred:
            super().update(y_true == y_pred)
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
        self.classes = set()

    def update(self, y_true, y_pred):
        self.precisions[y_pred].update(y_true == y_pred, True)
        self.classes.update({y_true, y_pred})
        return self

    def get(self):
        return statistics.mean((
            0. if c not in self.precisions else self.precisions[c].get()
            for c in self.classes
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

    def update(self, y_true, y_pred):
        super().update(y_true == y_pred)
        return self


class RollingPrecision(BasePrecision, base.BinaryMetric):
    """Rolling binary precision score.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingPrecision(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingPrecision: 1.
            RollingPrecision: 0.5
            RollingPrecision: 0.5
            RollingPrecision: 0.5
            RollingPrecision: 1.

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.tp_ratio = stats.RollingMean(window_size=window_size)
        self.fp_ratio = stats.RollingMean(window_size=window_size)

    def update(self, y_true, y_pred):
        self.tp_ratio.update(y_pred and y_true)
        self.fp_ratio.update(y_pred and not y_true)
        return self

    def get(self):
        tp = self.tp_ratio.get()
        fp = self.fp_ratio.get()
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0.


class RollingMacroPrecision(BasePrecision, base.MultiClassMetric):
    """Rolling macro-average precision score.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroPrecision(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMacroPrecision: 1.
            RollingMacroPrecision: 0.25
            RollingMacroPrecision: 0.5
            RollingMacroPrecision: 0.333333
            RollingMacroPrecision: 0.5

    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.rcm = confusion.RollingConfusionMatrix(window_size=window_size)

    def update(self, y_true, y_pred):
        self.rcm.update(y_true, y_pred)
        return self

    def get(self):

        # Use the rolling confusion matric to count the true positives and false positives
        classes = self.rcm.classes
        tps = collections.defaultdict(int)
        fps = collections.defaultdict(int)

        for yt, yp in itertools.product(classes, repeat=2):
            if yt == yp:
                tps[yp] = self.rcm.counts.get(yt, {}).get(yp, 0)
            else:
                fps[yp] += self.rcm.counts.get(yt, {}).get(yp, 0)

        def div_or_0(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.

        return statistics.mean((div_or_0(tps[c], tps[c] + fps[c]) for c in classes))


class RollingMicroPrecision(stats.RollingMean, BasePrecision, base.MultiClassMetric):
    """Rolling micro-average precision score.

    The micro-average precision score is exactly equivalent to the micro-average recall as well as
    the micro-average F1 score.

    Parameters:
        window_size (int): Size of the window of recent values to consider.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMicroPrecision(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMicroPrecision: 1.
            RollingMicroPrecision: 0.5
            RollingMicroPrecision: 0.666667
            RollingMicroPrecision: 0.666667
            RollingMicroPrecision: 0.666667

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """

    def update(self, y_true, y_pred):
        super().update(y_true == y_pred)
        return self
