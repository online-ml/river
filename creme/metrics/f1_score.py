import statistics

from .. import stats

from . import base
from . import precision
from . import recall


class F1Score(stats.Mean, base.BinaryClassificationMetric):
    """F1-score, which is the harmonic mean of the precision and the recall.

    Example:

        >>> from creme import metrics

        >>> y_true = [True, False, True, True, True]
        >>> y_pred = [True, True, False, True, True]

        >>> metric = metrics.F1Score()
        >>> for y_t, y_p in zip(y_true, y_pred):
        ...     print(metric.update(y_t, y_p).get())
        1.0
        0.666666...
        0.5
        0.666666...
        0.75

        >>> metric
        F1Score: 0.75

    """

    def __init__(self):
        super().__init__()
        self.precision = precision.Precision()
        self.recall = recall.Recall()

    @property
    def requires_labels(self):
        return True

    def update(self, y_true, y_pred):
        self.precision.update(y_true, y_pred)
        self.recall.update(y_true, y_pred)
        return self

    def get(self):
        return statistics.harmonic_mean((self.precision.get(), self.recall.get()))
