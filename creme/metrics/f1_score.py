import collections
import statistics

from .. import stats

from . import base
from . import precision
from . import recall


__all__ = ['F1Score', 'MacroF1Score', 'MicroF1Score']


class F1Score(stats.Mean, base.BinaryClassificationMetric):
    """Binary F1 score.

    The F1 score is the harmonic mean of the precision and the recall.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import f1_score

            >>> y_true = [True, False, True, True, True]
            >>> y_pred = [True, True, False, True, True]

            >>> metric = metrics.F1Score()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     assert metric.get() == f1_score(y_true[:i+1], y_pred[:i+1])

            >>> metric
            F1Score: 0.75

    """

    def __init__(self):
        super().__init__()
        self.precision = precision.Precision()
        self.recall = recall.Recall()

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def get(self):
        return statistics.harmonic_mean((self.precision.get(), self.recall.get()))


class MacroF1Score(base.MultiClassificationMetric):
    """Macro-average F1 score.

    The macro-average F1 score is the arithmetic average of the binary F1 scores of each label.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import f1_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MacroF1Score()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), f1_score(y_true[:i+1], y_pred[:i+1], average='macro'))
            1.0 1.0
            0.333333... 0.333333...
            0.555555... 0.555555...
            0.555555... 0.555555...
            0.488888... 0.488888...

            >>> metric
            MacroF1Score: 0.488889

    """

    def __init__(self):
        self.f1_scores = collections.defaultdict(F1Score)
        self.classes = set()

    @property
    def bigger_is_better(self):
        return True

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        self.classes.update({y_true, y_pred})
        for c in self.classes:
            self.f1_scores[c].update(y_true == c, y_pred == c)
        return self

    def get(self):
        total = sum(f1.get() for f1 in self.f1_scores.values())
        if total == 0.:
            return 0.
        return total / len(self.f1_scores)


class MicroF1Score(precision.MicroPrecision):
    """Micro-average F1 score.

    The micro-average F1 score is exactly equivalent to the micro-average precision as well as the
    micro-average recall score.

    Example:

        ::

            >>> from creme import metrics
            >>> from sklearn.metrics import f1_score

            >>> y_true = [0, 1, 2, 2, 2]
            >>> y_pred = [0, 0, 2, 2, 1]

            >>> metric = metrics.MicroF1Score()
            >>> for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
            ...     metric = metric.update(y_t, y_p)
            ...     print(metric.get(), f1_score(y_true[:i+1], y_pred[:i+1], average='micro'))
            1.0 1.0
            0.5 0.5
            0.666666... 0.666666...
            0.75 0.75
            0.6 0.6

            >>> metric
            MicroF1Score: 0.6

    References:

        1. `Why are precision, recall and F1 score equal when using micro averaging in a multi-class problem? <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>`_

    """
