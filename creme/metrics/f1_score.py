import collections
import functools
import statistics

from . import base
from . import precision
from . import recall


__all__ = [
    'F1Score',
    'MacroF1Score',
    'MicroF1Score',
    'RollingF1Score',
    'RollingMacroF1Score',
    'RollingMicroF1Score'
]


class BaseF1Score:

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True


class F1Score(BaseF1Score, base.BinaryMetric):
    """Binary F1 score.

    The F1 score is the harmonic mean of the precision and the recall.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.F1Score()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            F1Score: 1.
            F1Score: 0.666667
            F1Score: 0.5
            F1Score: 0.666667
            F1Score: 0.75

    """

    def __init__(self):
        super().__init__()
        self.precision = precision.Precision()
        self.recall = recall.Recall()

    def update(self, y_true, y_pred):
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def get(self):
        return statistics.harmonic_mean((self.precision.get(), self.recall.get()))


class MacroF1Score(BaseF1Score, base.MultiClassMetric):
    """Macro-average F1 score.

    The macro-average F1 score is the arithmetic average of the binary F1 scores of each label.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroF1Score()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MacroF1Score: 1.
            MacroF1Score: 0.333333
            MacroF1Score: 0.555556
            MacroF1Score: 0.555556
            MacroF1Score: 0.488889

    """

    def __init__(self):
        self.f1_scores = collections.defaultdict(F1Score)
        self.classes = set()

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})
        for c in self.classes:
            self.f1_scores[c].update(y_true == c, y_pred == c)
        return self

    def get(self):
        total = sum(f1.get() for f1 in self.f1_scores.values())
        try:
            return total / len(self.f1_scores)
        except ZeroDivisionError:
            return 0.


class MicroF1Score(precision.MicroPrecision):
    """Micro-average F1 score.

    The micro-average F1 score is exactly equivalent to the micro-average precision as well as the
    micro-average recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroF1Score()

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            MicroF1Score: 1.
            MicroF1Score: 0.5
            MicroF1Score: 0.666667
            MicroF1Score: 0.75
            MicroF1Score: 0.6

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """


class RollingF1Score(F1Score):
    """Rolling binary F1 score.

    The F1 score is the harmonic mean of the precision and the recall.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.RollingF1Score(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingF1Score: 1.
            RollingF1Score: 0.666667
            RollingF1Score: 0.5
            RollingF1Score: 0.5
            RollingF1Score: 0.8

    """

    def __init__(self, window_size):
        super().__init__()
        self.precision = precision.RollingPrecision(window_size=window_size)
        self.recall = recall.RollingRecall(window_size=window_size)

    @property
    def window_size(self):
        return self.precision.window_size


class RollingMacroF1Score(MacroF1Score):
    """Rolling macro-average F1 score.

    The macro-average F1 score is the arithmetic average of the binary F1 scores of each label.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMacroF1Score(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp))
            RollingMacroF1Score: 1.
            RollingMacroF1Score: 0.333333
            RollingMacroF1Score: 0.555556
            RollingMacroF1Score: 0.333333
            RollingMacroF1Score: 0.266667

    """

    def __init__(self, window_size):
        self.f1_scores = collections.defaultdict(functools.partial(RollingF1Score, window_size))
        self.classes = set()


class RollingMicroF1Score(precision.RollingMicroPrecision):
    """Rolling micro-average F1 score.

    The micro-average F1 score is exactly equivalent to the micro-average precision as well as the
    micro-average recall score.

    Example:

        ::

            >>> from creme import metrics

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.RollingMicroF1Score(window_size=3)

            >>> for yt, yp in zip(y_true, y_pred):
            ...     print(metric.update(yt, yp).get())
            1.0
            0.5
            0.666666...
            0.666666...
            0.666666...

            >>> metric
            RollingMicroF1Score: 0.666667

    References:
        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """
