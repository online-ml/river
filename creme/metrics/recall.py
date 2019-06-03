import collections
import itertools
import statistics

from .. import stats

from . import base
from . import confusion
from . import precision


__all__ = [
    'MacroRecall',
    'MicroRecall',
    'Recall',
    'RollingMacroRecall',
    'RollingMicroRecall',
    'RollingRecall'
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

    def update(self, y_true, y_pred):
        if y_true:
            return super().update(y_true == y_pred)
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
        self.classes = set()

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


class RollingRecall(BaseRecall, base.BinaryMetric):
    """Rolling binary recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingRecall(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingRecall: 1.
            RollingRecall: 1.
            RollingRecall: 0.5
            RollingRecall: 0.5
            RollingRecall: 0.666667

    """

    def __init__(self, window_size):
        self.tp_ratio = stats.RollingMean(window_size=window_size)
        self.fn_ratio = stats.RollingMean(window_size=window_size)

    @property
    def window_size(self):
        return self.tp_ratio.size

    def update(self, y_true, y_pred):
        self.tp_ratio.update(y_true and y_pred)
        self.fn_ratio.update(y_true and not y_pred)
        return self

    def get(self):
        tp = self.tp_ratio.get()
        fn = self.fn_ratio.get()
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0.


class RollingMacroRecall(MacroRecall):
    """Rolling macro-average recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroRecall(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMacroRecall: 1.
            RollingMacroRecall: 0.5
            RollingMacroRecall: 0.666667
            RollingMacroRecall: 0.333333
            RollingMacroRecall: 0.333333

    """

    def __init__(self, window_size):
        self.rcm = confusion.RollingConfusionMatrix(window_size=window_size)

    @property
    def window_size(self):
        return self.rcm.window_size

    def update(self, y_true, y_pred):
        self.rcm.update(y_true, y_pred)
        return self

    def get(self):

        # Use the rolling confusion matric to count the true positives and false negatives
        classes = self.rcm.classes
        tps = collections.defaultdict(int)
        fns = collections.defaultdict(int)

        for yt, yp in itertools.product(classes, repeat=2):
            if yt == yp:
                tps[yp] = self.rcm.get(yt, {}).get(yp, 0)
            else:
                fns[yp] += self.rcm.get(yp, {}).get(yt, 0)

        def div_or_0(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.

        return statistics.mean((div_or_0(tps[c], tps[c] + fns[c]) for c in classes))


class RollingMicroRecall(precision.RollingMicroPrecision):
    """Rolling micro-average recall score.

    The micro-average recall is exactly equivalent to the micro-average precision as well as the
    micro-average F1 score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMicroRecall(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMicroRecall: 1.
            RollingMicroRecall: 0.5
            RollingMicroRecall: 0.666667
            RollingMicroRecall: 0.666667
            RollingMicroRecall: 0.666667

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """
